package http2push

import (
	"log"
	"os"
)

var (
	// ServerLogger is the logger for the server
	ServerLogger = log.New(os.Stderr, "[server] ", log.LstdFlags)
	// RunnerLogger is the logger for the runner
	RunnerLogger = log.New(os.Stderr, "[runner] ", log.LstdFlags)
)
