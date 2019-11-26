#! /usr/bin/env node

const child_process = require("child_process")

const commandLineArgs = require("command-line-args");

const utils = require("./utils");
const serverArguments = require("./server/args");
const captureHarArguments = require("./capturer/args");

const argumentsDefinition = [
  ...serverArguments,
  ...captureHarArguments,
  { name: 'link-trace-path', alias: 't', defaultValue: '', type: String },
  { name: 'link-latency-ms', alias: 'l', defaultValue: 0, type: Number },
  { name: 'user-id', alias: 'u', defaultValue: 0, type: Number },
  { name: 'group-id', alias: 'g', defaultValue: 0, type: Number },
  { name: 'force-stop', defaultValue: false, type: Boolean },
];

const run = async args => {
  // create and start the server
  let exitCode = -1;
  const replayArgs = [
    "replay",
    "--cert_path", args.certFile,
    "--key_path", args.keyFile,
    "--policy", args.policyPath,
    args.fileStorePath,
  ];
  const server = child_process.spawn("blaze", replayArgs, { stdio: 'inherit' });
  server.on('exit', code => { exitCode = code });
  console.log("starting replay server with args", replayArgs);

  // wait a few seconds and make sure the server is up
  await utils.asyncWait(5000);
  if (exitCode !== -1) {
    console.error("replay server failed to start");
    process.exit(1);
  }

  const captureCmd = [];
  if (args.linkTracePath)
    captureCmd.push("mm-link", args.linkTracePath, args.linkTracePath, "--");
  if (args.linkLatencyMs > 0)
    captureCmd.push("mm-delay", args.linkLatencyMs.toString());
    if (args.extractCriticalRequests)
    captureCmd.push("sudo", "npm", "run", "capturer", "--", "-o", args.outputFile, "-s", args.cpuSlowdown, args.url, "-d", args.userDataDir, "-x");
  else 
    captureCmd.push("sudo", "npm", "run", "capturer", "--", "-o", args.outputFile, "-s", args.cpuSlowdown, args.url, "-d", args.userDataDir);

  await utils.run(captureCmd, args.userId, args.groupId);
  console.log("Finished capturing HAR...");

  server.kill('SIGKILL');
  if (args.forceStop) {
    process.exit(0);
  }
};

const main = async args => {
  try {
    console.log("Starting with args", args);
    await run(args);
  } catch (e) {
    console.error(e);
  }
};

main(commandLineArgs(argumentsDefinition, { camelCase: true }));
