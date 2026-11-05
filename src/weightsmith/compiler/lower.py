"""Lower a type-checked program to IR."""

from __future__ import annotations

from typing import Any

from weightsmith.compiler import ir
from weightsmith.compiler.typecheck import Binding, CheckedFile, CheckedProgram
from weightsmith.language import nodes as n


class _Lowerer:
    def __init__(self, file: CheckedFile, program: CheckedProgram) -> None:
        self.file = file
        self.program = program
        self.ops: list[ir.Op] = []
        self.values: dict[str, int] = {}  # binding name -> value id
        self.probe_sizes: dict[int, int] = {}
        self.next_id = 0
        self.env: dict[str, Binding] = {**file.bindings, **program.locals}

    def fresh(self) -> int:
        value = self.next_id
        self.next_id += 1
        return value

    def load(self, binding: Binding) -> int:
        if binding.name in self.values:
            return self.values[binding.name]
        out = self.fresh()
        assert binding.path is not None
        params: dict[str, Any] = dict(binding.params)
        self.ops.append(
            ir.Load(out, binding.name, binding.kind, str(binding.path), params, span=binding.span)
        )
        if binding.suite is not None:
            self.probe_sizes[out] = binding.suite.n
        self.values[binding.name] = out
        return out

    def expr(self, expr: n.Expr) -> int:
        role = self.file.roles.get(id(expr))
        if role is not None:
            binding, ref = role
            source = self.load(self.env[binding])
            out = self.fresh()
            self.ops.append(ir.Select(out, source, role=str(ref), span=expr.span))
            return out
        match expr:
            case n.Name():
                binding = self.env[expr.ident]
                if binding.kind == "selection":
                    if binding.name not in self.values:
                        assert binding.expr is not None
                        self.values[binding.name] = self.expr(binding.expr)
                    return self.values[binding.name]
                return self.load(binding)
            case n.Subscript(obj=n.Attr(obj=n.Name() as root, attr=attr)) if attr.name == "params":
                pattern = expr.indices[0]
                assert isinstance(pattern, n.StrLit)
                source = self.load(self.env[root.ident])
                out = self.fresh()
                self.ops.append(ir.Select(out, source, pattern.value, span=expr.span))
                return out
            case n.Subscript():
                source = self.expr(expr.obj)
                out = self.fresh()
                self.ops.append(ir.Select(out, source, None, _index(expr.indices), span=expr.span))
                return out
            case _:  # pragma: no cover - rejected by the type checker
                raise AssertionError(f"cannot lower expression {expr!r}")

    def kwargs(self, node: object) -> dict[str, Any]:
        return dict(self.program.kwargs.get(id(node), {}))

    def lower(self) -> ir.IRProgram:
        program = self.program.ast
        target = self.load(self.program.target)
        contract_items = self.contract()
        for stmt in program.body:
            match stmt:
                case n.LetStmt():
                    self.values[stmt.name.name] = self.expr(stmt.expr)
                case n.InspectStmt():
                    source = self.expr(stmt.subject) if stmt.subject is not None else target
                    self.ops.append(ir.Inspect(source, span=stmt.span))
                case n.FindStmt():
                    scope = self.expr(stmt.scope) if stmt.scope is not None else target
                    self.ops.append(
                        ir.Find(stmt.finder.name, scope, self.kwargs(stmt), span=stmt.span)
                    )
                case n.EditBlock():
                    for op in stmt.ops:
                        operand = self.expr(op.operand)
                        self.ops.append(
                            ir.Transform(op.op.name, operand, self.kwargs(op), span=op.span)
                        )
                case n.ComposeBlock():
                    for op in stmt.ops:
                        operand = self.expr(op.operand)
                        self.ops.append(
                            ir.Compose(op.op.name, operand, self.kwargs(op), span=op.span)
                        )
                case n.RepairStmt():
                    strategies = tuple(self.kwargs(stmt).get("strategies", ()))
                    self.ops.append(ir.Repair(strategies, span=stmt.span))
                case n.DelegateStmt():
                    self.ops.append(self.delegate(stmt))
                case n.VerifyStmt():
                    self.ops.append(ir.Verify(contract_items, span=stmt.span))
                case n.OutputBlock():
                    self.ops.append(ir.Emit(tuple(i.name for i in stmt.items), span=stmt.span))
                case n.ContractBlock() | n.PolicyBlock():
                    pass
        return ir.IRProgram(
            name=program.name.name,
            source=self.file.ast.path,
            target=target,
            target_binding=self.program.target.name,
            target_is_model=self.program.target.kind == "model",
            policy=self.program.policy,
            ops=tuple(self.ops),
            probe_sizes=self.probe_sizes,
            span=program.span,
        )

    def delegate(self, stmt: n.DelegateStmt) -> ir.Delegate:
        values = self.kwargs(stmt)
        editor = self.env[stmt.editor.name]
        settings = dict(editor.params)
        provider = str(settings.pop("provider"))
        practice = tuple(self.load(self.file.bindings[b]) for b in values.get("practice", ()))
        keys = (
            "max_steps",
            "max_read",
            "include_weights",
            "protocol",
            "system",
            "message",
            "weights_format",
        )
        params = {k: values[k] for k in keys if k in values}
        return ir.Delegate(
            editor.name,
            provider,
            settings,
            str(values.get("task", "")),
            tuple(values.get("tools", ())),
            practice,
            params,
            span=stmt.span,
        )

    def contract(self) -> tuple[ir.ContractItem, ...]:
        block = self.program.contract
        if block is None:
            return ()
        uses_verify = any(isinstance(s, n.VerifyStmt) for s in self.program.ast.body)
        items = []
        for item in block.items:
            behavior = None
            if item.clause != "require" and uses_verify:
                behavior = self.load(self.file.bindings[item.subject.name])
            items.append(
                ir.ContractItem(
                    item.clause, item.subject.name, behavior, self.kwargs(item), item.span
                )
            )
        return tuple(items)


def _index(indices: tuple[n.Index, ...]) -> tuple[ir.IndexSpec, ...]:
    out = []
    for index in indices:
        match index:
            case n.IntLit():
                out.append(ir.IndexSpec("int", index.value, index.value + 1))
            case n.RangeIndex():
                out.append(ir.IndexSpec("range", index.start, index.stop))
            case n.FullIndex():
                out.append(ir.IndexSpec("all"))
            case _:  # pragma: no cover
                raise AssertionError(index)
    return tuple(out)


def lower_program(file: CheckedFile, program: CheckedProgram) -> ir.IRProgram:
    return _Lowerer(file, program).lower()
