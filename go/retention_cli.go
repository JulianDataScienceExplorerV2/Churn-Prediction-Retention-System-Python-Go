// retention_cli.go — Churn Risk Action List Generator
// ====================================================
// EN: CLI tool to read churn probabilities from the Python model,
//     filter high-risk customers, and generate a daily call list.
// 
// ES: Herramienta CLI para leer probabilidades del modelo Python,
//     filtrar clientes de alto riesgo y generar una lista diaria de llamadas.
package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"os"
	"sort"
	"strconv"
)

// ANSI color codes
const (
	reset  = "\033[0m"
	bold   = "\033[1m"
	red    = "\033[31m"
	yellow = "\033[33m"
	green  = "\033[32m"
	cyan   = "\033[36m"
)

type CustomerRisk struct {
	ID           string
	Tenure       int
	SupportCalls int
	RiskProb     float64
	ChurnActual  int
}

func main() {
	var filePath = flag.String("file", "../data/churn_predictions.csv", "Path to the predictions CSV")
	var threshold = flag.Float64("threshold", 0.70, "Minimum probability (0.0 - 1.0) to flag as At-Risk")
	var limit = flag.Int("limit", 15, "Max number of customers to show in the action list")
	flag.Parse()

	f, err := os.Open(*filePath)
	if err != nil {
		fmt.Printf("%sFailed to open %s: %v%s\n", red, *filePath, err, reset)
		os.Exit(1)
	}
	defer f.Close()

	reader := csv.NewReader(f)
	rows, err := reader.ReadAll()
	if err != nil {
		fmt.Printf("%sCSV parsing error: %v%s\n", red, err, reset)
		os.Exit(1)
	}

	var atRisk []CustomerRisk
	var totalCustomers = len(rows) - 1
	var totalChurned = 0
	var truePositives = 0

	for i, row := range rows {
		if i == 0 {
			continue // skip header
		}
		
		id := row[0]
		tenure, _ := strconv.Atoi(row[1])
		support, _ := strconv.Atoi(row[3])
		prob, _ := strconv.ParseFloat(row[4], 64)
		actual, _ := strconv.Atoi(row[5])

		if actual == 1 {
			totalChurned++
		}

		if prob >= *threshold {
			atRisk = append(atRisk, CustomerRisk{
				ID:           id,
				Tenure:       tenure,
				SupportCalls: support,
				RiskProb:     prob,
				ChurnActual:  actual,
			})
			if actual == 1 {
				truePositives++
			}
		}
	}

	// Sort explicitly by highest risk
	sort.Slice(atRisk, func(i, j int) bool {
		return atRisk[i].RiskProb > atRisk[j].RiskProb
	})

	// Print Actionable Output
	fmt.Printf("\n%s%sDAILY RETENTION ACTION LIST%s\n", cyan, bold, reset)
	fmt.Printf("Model Threshold: > %.0f%% Risk Probability\n", *threshold*100)
	fmt.Printf("Total Customers Analyzed: %d\n\n", totalCustomers)

	fmt.Printf("%s%-15s %-12s %-15s %-12s %-15s%s\n", bold, "CUSTOMER ID", "TENURE (mo)", "SUPPORT CALLS", "RISK PROB", "RECOMMENDATION", reset)
	fmt.Printf("----------------------------------------------------------------------------\n")

	displayCount := *limit
	if len(atRisk) < displayCount {
		displayCount = len(atRisk)
	}

	for i := 0; i < displayCount; i++ {
		c := atRisk[i]
		
		// Color code risk: >90% = red, >80% = yellow
		color := reset
		if c.RiskProb >= 0.90 {
			color = red + bold
		} else if c.RiskProb >= 0.80 {
			color = yellow
		}
		
		// Action logic
		action := "Offer 10% Discount"
		if c.SupportCalls >= 4 {
			action = "Agent Call ASAP"
		} else if c.Tenure <= 3 {
			action = "Onboarding Flow"
		}

		fmt.Printf("%s%-15s %-12d %-15d %-11s %-15s%s\n", 
			color, c.ID, c.Tenure, c.SupportCalls, fmt.Sprintf("%.1f%%", c.RiskProb*100), action, reset)
	}

	fmt.Printf("----------------------------------------------------------------------------\n")
	fmt.Printf("Found %s%d customers%s needing immediate retention action.\n\n", yellow, len(atRisk), reset)
}
