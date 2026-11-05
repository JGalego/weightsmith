// Starts the Weightsmith language server (`weightsmith lsp`) for .wsm files.
"use strict";

const fs = require("fs");
const path = require("path");
const vscode = require("vscode");
const { LanguageClient } = require("vscode-languageclient/node");

let client;

// The configured command, else the workspace's virtual environment, else `weightsmith` on PATH.
function serverCommand() {
  const configured = vscode.workspace.getConfiguration("weightsmith").get("server.command");
  if (Array.isArray(configured) && configured.length > 0) {
    return { command: configured[0], args: configured.slice(1) };
  }
  const bin = process.platform === "win32" ? path.join("Scripts", "weightsmith.exe") : path.join("bin", "weightsmith");
  for (const folder of vscode.workspace.workspaceFolders || []) {
    const candidate = path.join(folder.uri.fsPath, ".venv", bin);
    if (fs.existsSync(candidate)) {
      return { command: candidate, args: [] };
    }
  }
  return { command: "weightsmith", args: [] };
}

async function activate(context) {
  const { command, args } = serverCommand();
  const run = { command, args: [...args, "lsp"] };
  client = new LanguageClient(
    "weightsmith",
    "Weightsmith",
    { run, debug: run },
    { documentSelector: [{ scheme: "file", language: "weightsmith" }, { scheme: "untitled", language: "weightsmith" }] }
  );
  try {
    await client.start();
  } catch (error) {
    vscode.window.showWarningMessage(
      `Weightsmith: could not start the language server with '${command}'. ` +
        "Install it with pip install 'weightsmith[lsp]', or set weightsmith.server.command."
    );
  }
  context.subscriptions.push(client);
}

function deactivate() {
  return client ? client.stop() : undefined;
}

module.exports = { activate, deactivate };
