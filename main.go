package main

import (
	"fmt"
	"math"
	"tp/concurrent/ccrandomforest"
	"tp/csvreader"
	"tp/sequential/sqrandomforest"
	"tp/test"

	//"pc2/concurrent/ccsvm"
	//"pc2/sequential/sqsvm"

	"time"
)

func readCsvAsMatrix(path string) [][]float64 {
	_, textData, err := csvreader.ReadCSV(path)
	if err != nil {
		fmt.Println("Error:", err)
		return nil
	}
	floatData, err := csvreader.ConvertToFloat64(textData)
	if err != nil {
		fmt.Println("Error:", err)
		return nil
	}
	return floatData
}
func matrixToSlice(matrix [][]float64) []float64 {
	slice := make([]float64, len(matrix))
	//fmt.Println(matrix[:1000])

	for i, row := range matrix {
		slice[i] = row[0]
	}
	return slice
}

func main() {
	var trainStart, testStart time.Time
	var trainDuration, testDuration time.Duration
	var predY []float64

	xTrain := readCsvAsMatrix("data/X_train.csv")
	yTrain := matrixToSlice(readCsvAsMatrix("data/y_train.csv"))
	xTest := readCsvAsMatrix("data/X_test.csv")
	yTest := matrixToSlice(readCsvAsMatrix("data/y_test.csv"))

	if xTrain == nil || yTrain == nil || xTest == nil || yTest == nil {
		return
	}
	println("X train -> rows  =", len(xTrain), "\tcols =", len(xTrain[0]))
	println("y train -> nrows =", len(yTrain))
	println("X test  -> rows  =", len(xTest), "\tcols =", len(xTest[0]))
	println("y test  -> nrows =", len(yTest))
	fmt.Println(":::::::::::::::::::::::::::::::::::::::::")
	fmt.Println("::::::: RANDOM FOREST CLASSIFIER ::::::::")
	fmt.Println(":::::::::::::::::::::::::::::::::::::::::")
	// Random Forest parameters
	// seqRfModel := sqrf.RandomForest{NumTrees: 200, MaxDepth: 20, MaxFeatures: int(math.Sqrt(float64(len(X[0]))))}
	numTrees := 200
	maxDepth := 30
	maxFeatures := int(math.Sqrt(float64(len(xTrain[0]))))
	// Random Forest threads
	treeThreads := 10
	featureThreads := 10

	fmt.Println("NumTrees:       ", numTrees)
	fmt.Println("MaxDepth:       ", maxDepth)
	fmt.Println("MaxFeatures:    ", maxFeatures)
	fmt.Println("TreeThreads:    ", treeThreads)
	fmt.Println("FeatureThreads: ", featureThreads)

	fmt.Println("-----------------------------------------")
	fmt.Println("\tConcurrent")
	fmt.Println("-----------------------------------------")
	/*ccRfModel := ccrandomforest.RandomForest{
		NumTrees:    numTrees,
		MaxDepth:    maxDepth,
		MaxFeatures: maxFeatures,
	}*/
	ccRfModel := ccrandomforest.NewRandomForestClassifier(numTrees, maxDepth, maxFeatures, 100)
	trainStart = time.Now()
	ccRfModel.Train(xTrain, yTrain, treeThreads, featureThreads)
	trainDuration = time.Since(trainStart)

	testStart = time.Now()
	predY = ccRfModel.Predict(xTest)
	testDuration = time.Since(testStart)

	test.ConfusionMatrix(yTest, predY)
	fmt.Println("-Results-")
	fmt.Println("\tpresicion : ", test.CalcPrecision(yTest, predY))
	fmt.Println("\trecall    : ", test.CalcRecall(yTest, predY))
	fmt.Println("\tf1-score  : ", test.CalcF1Score(yTest, predY))
	fmt.Printf("Training duration: %v\n", trainDuration)
	fmt.Printf("Predict duration : %v\n", testDuration)
	fmt.Println("-----------------------------------------")
	fmt.Println("\tSequential")
	fmt.Println("-----------------------------------------")
	/*seqRfModel := sqrandomforest.RandomForest{
		NumTrees:    numTrees,
		MaxDepth:    maxDepth,
		MaxFeatures: maxFeatures,
	}*/
	seqRfModel := sqrandomforest.NewRandomForestClassifier(numTrees, maxDepth, maxFeatures, 100)

	trainStart = time.Now()
	seqRfModel.Train(xTrain, yTrain)
	trainDuration = time.Since(trainStart)

	testStart = time.Now()
	predY = seqRfModel.Predict(xTest)
	testDuration = time.Since(testStart)

	test.ConfusionMatrix(yTest, predY)
	fmt.Println("-Results-")
	fmt.Println("\tpresicion : ", test.CalcPrecision(yTest, predY))
	fmt.Println("\trecall    : ", test.CalcRecall(yTest, predY))
	fmt.Println("\tf1-score  : ", test.CalcF1Score(yTest, predY))
	fmt.Printf("Training duration: %v\n", trainDuration)
	fmt.Printf("Predict duration : %v\n", testDuration)
}
