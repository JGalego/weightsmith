# The README's terminal recording. Regenerate with `just demo` (needs vhs and the fixtures).
# It runs in a scratch directory so the recording shows no local paths.
Output assets/demo.gif

Set Shell "bash"
Set FontSize 14
Set Width 1100
Set Height 820
Set Padding 20
Set Theme "Catppuccin Mocha"
Set TypingSpeed 35ms
Set WaitTimeout 5m

Hide
Type `export REPO=$PWD DEMO=/tmp/weightsmith-demo && rm -rf $DEMO && mkdir -p $DEMO/examples/fixtures` Enter
Type `cp -r $REPO/examples/repair_with_editor.wsm $REPO/examples/editors $DEMO/examples/` Enter
Type `cp -r $REPO/examples/fixtures/tiny_llama $REPO/examples/fixtures/tiny_llama_defective $DEMO/examples/fixtures/` Enter
Type `source $REPO/.venv/bin/activate && cd $DEMO && clear` Enter
Show

Type "weightsmith check examples/repair_with_editor.wsm" Sleep 300ms Enter
Wait+Screen /Valid/
Sleep 2s
Type "weightsmith run examples/repair_with_editor.wsm --out out" Sleep 300ms Enter
Wait+Screen /Completed/
Sleep 8s

Hide
Type `cd $REPO && rm -rf $DEMO` Enter
