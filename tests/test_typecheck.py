from __future__ import annotations

from collections.abc import Callable

import pytest

from weightsmith.compiler.driver import CompileResult

from .conftest import HEADER, codes

Compile = Callable[[str], CompileResult]


def program(body: str, target: str = "Base", extra: str = "") -> str:
    policy = "policy { allow probe allow mutate allow save }"
    return f"{HEADER}\n{extra}\nprogram P {{\n target {target}\n {policy}\n {body}\n}}\n"


def test_valid_program_has_no_diagnostics(compile_text: Compile) -> None:
    result = compile_text(
        program("""
        contract { improve parity (min_delta = 0.1) preserve majority (tolerance = 0.02) }
        find anomalies (kinds = [outlier_rows])
        repair (strategies = [rescale_outlier_rows])
        verify
        output { model, patch, findings, evaluation }
    """)
    )
    assert result.ok, [d.message for d in result.sink.items]
    assert not result.sink.items


@pytest.mark.parametrize(
    ("body", "code"),
    [
        ("inspect Nope", "E201"),
        ("find reasoning_loops", "E205"),
        ('edit { remove Base.params["head.weight"] }', "E206"),
        ("compose { zero Donor }", "E207"),
        ("contract { require tidy } verify", "E208"),
        ("output { explanation }", "E209"),
        ('let Base = Base.params["head.weight"]', "E204"),
        ("inspect Base.weights", "E310"),
        ("inspect Base.params[0]", "E312"),
        ('inspect Base.params["head.weight"][0][1]', "E313"),
        ("find outlier_rows (threshold = 6.0, threshold = 7.0)", "E320"),
        ("find outlier_rows (cutoff = 3.0)", "E321"),
        ('edit { scale Base.params["head.weight"] }', "E322"),
        ("find anomalies (kinds = [])", "E323"),
        ("find anomalies (kinds = [anomalies])", "E324"),
        ('edit { scale Base.params["head.weight"] (factor = "big") }', "E326"),
        ('edit { prune_magnitude Base.params["head.weight"] (fraction = 150%) }', "E327"),
        (
            "contract { preserve parity (tolerance = 0.1) } contract { require finite } verify",
            "E331",
        ),
        ("repair (strategies = [rescale_outlier_rows])", "E340"),
        ("find nan_inf repair (strategies = [rescale_outlier_rows])", "E341"),
        ("verify", "E342"),
        ('find nan_inf in Base.params["head.weight"][0]', "E343"),
        ('edit { zero Donor.params["head.weight"] }', "E346"),
        ('edit { low_rank Base.params["head.weight"][0] (rank = 1) }', "E347"),
        ("compose { interpolate Base (alpha = 0.5) }", "E348"),
        ("contract { preserve Donor (tolerance = 0.1) } verify", "E351"),
        ("output { model }", "E353"),
        ("output { evaluation }", "E353"),
        ('inspect Base.params["nothing.*"]', "E501"),
        ('edit { zero Base.params["layers.1.weight"][64, :] }', "E503"),
        ('edit { zero Base.params["layers.1.bias"][0, 0] }', "E502"),
        ('edit { low_rank Base.params["layers.1.bias"] (rank = 1) }', "E504"),
        ("output { summary } inspect", "E332"),
    ],
)
def test_static_errors(compile_text: Compile, body: str, code: str) -> None:
    result = compile_text(program(body))
    assert code in codes(result), [(d.code, d.message) for d in result.sink.items]


def test_each_error_is_reported_once(compile_text: Compile) -> None:
    result = compile_text(program('edit { zero Base.params["layers.1.weight"][70, :] }'))
    assert codes(result) == ["E503"]


def test_unknown_symbol_does_not_also_report_missing_argument(compile_text: Compile) -> None:
    result = compile_text(
        'behavior b = probes("fixtures/tiny_mlp/parity.json", metric = acuracy)'
        '\nweights W = load("fixtures/tiny_mlp/model.safetensors")'
        "\nprogram P { target W inspect }"
    )
    assert codes(result) == ["E203"]
    assert any("accuracy" in note for d in result.sink.errors for note in d.notes)


def test_behaviour_checks_need_a_model(compile_text: Compile) -> None:
    result = compile_text(
        program("contract { preserve parity (tolerance = 0.1) } verify", target="W")
    )
    assert "E352" in codes(result)


def test_weights_cannot_be_loaded_as_model(compile_text: Compile) -> None:
    result = compile_text(
        'model M = load("fixtures/tiny_mlp/model.safetensors")\nprogram P { target M inspect }'
    )
    assert "E304" in codes(result)


def test_directory_without_arch_is_not_a_model(compile_text: Compile) -> None:
    result = compile_text(
        'model M = load("fixtures/tiny_transformer")\nprogram P { target M inspect }'
    )
    assert "E304" in codes(result)


def test_interpolation_needs_matching_manifests(compile_text: Compile) -> None:
    result = compile_text(
        program(
            "compose { interpolate T (alpha = 0.5) }",
            extra='weights T = load("fixtures/tiny_transformer/model.safetensors")',
        )
    )
    assert "E505" in codes(result)


def test_transplant_checks_names(compile_text: Compile) -> None:
    result = compile_text(
        program(
            'compose { transplant T.params["embed.weight"] }',
            extra='weights T = load("fixtures/tiny_transformer/model.safetensors")',
        )
    )
    assert "E506" in codes(result)


def test_missing_files_are_warnings_not_errors(compile_text: Compile) -> None:
    result = compile_text(
        'model M = load("nowhere")\nprogram P { target M inspect M.params["anything"] }'
    )
    assert result.ok
    assert [d.code for d in result.sink.warnings] == ["W201"]


def test_threshold_finer_than_resolution_warns(compile_text: Compile) -> None:
    result = compile_text(program("contract { improve parity (min_delta = 0.001) } verify"))
    assert "W303" in [d.code for d in result.sink.warnings]


def test_fractions_accept_plain_numbers_in_range(compile_text: Compile) -> None:
    result = compile_text(
        program('edit { prune_magnitude Base.params["head.weight"] (fraction = 0.3) }')
    )
    assert result.ok


def test_selection_declarations_and_lets(compile_text: Compile) -> None:
    result = compile_text(
        program(
            "let rows = hidden[0..4, :]\n edit { zero rows }\n output { patch }",
            extra='selection hidden = Base.params["layers.1.weight"]',
        )
    )
    assert result.ok, [d.message for d in result.sink.items]
