package main

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"strconv"
	"tp/concurrent/ccrandomforest"
	"tp/csvreader"
)

func readCsvAsMatrix(path string, Comma rune) [][]float64 {
	_, textData, err := csvreader.ReadCSV(path, Comma)
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

func getModel(numTrees, maxDepth int) ccrandomforest.RandomForest {
	/*numTrees := 200
	maxDepth := 30
	*/
	xTrain := readCsvAsMatrix("data/X_train.csv", ';')
	yTrain := matrixToSlice(readCsvAsMatrix("data/y_train.csv", ';'))
	treeThreads := 10
	featureThreads := 10
	maxFeatures := int(math.Sqrt(float64(len(xTrain[0]))))
	rfc := ccrandomforest.NewRandomForestClassifier(numTrees, maxDepth, maxFeatures, 100)
	rfc.Train(xTrain, yTrain, treeThreads, featureThreads)
	return rfc
}
func readInput(prompt string) string {
	reader := bufio.NewReader(os.Stdin)
	fmt.Print(prompt)
	input, _ := reader.ReadString('\n')
	return input[:len(input)-1]
}

func get_encoder(pathfile string) map[string]float64 {
	_, textData, err := csvreader.ReadCSV(pathfile, ',')
	if err != nil {
		fmt.Println("Error:", err)
	}
	encoder := make(map[string]float64)
	for _, row := range textData {
		encoder[row[0]], _ = strconv.ParseFloat(row[1], 64)
	}
	return encoder
}

func printEncoderCodes(encoder map[string]float64) {
	for key := range encoder {
		fmt.Print(key + " ")
	}
	fmt.Println()
}

func valideEncoder(encoder map[string]float64, value string) bool {
	_, ok := encoder[value]
	return ok
}

func banner(name string) {
	fmt.Println(":::::::::::::::::::::::::::::::::::::::::")
	fmt.Println("::::::: ", name, " ::::::::")
	fmt.Println(":::::::::::::::::::::::::::::::::::::::::")
}

func read_valid_input(encoder map[string]float64, prompt string) string {
	info := "i"
	fmt.Println("Para ver los valores válidos ingrese ", info)
	value := readInput(prompt)
	if value == info {
		printEncoderCodes(encoder)
	}
	for !valideEncoder(encoder, value) {
		if value != info {
			fmt.Println("Valor no válido")
			fmt.Println("Para ver los valores válidos ingrese ", info)
		}
		value = readInput(prompt)
		if value == info {
			printEncoderCodes(encoder)
		}
	}
	return value
}

func app() {
	rfc := getModel(200, 30)
	encoderDepartamento := get_encoder("data/encoder/Departamento.csv")
	encoderProvincia := get_encoder("data/encoder/Provincia.csv")
	encoderDistrito := get_encoder("data/encoder/Distrito.csv")
	encoderCartera := get_encoder("data/encoder/Cartera.csv")
	encoderTarifa := get_encoder("data/encoder/Tarifa.csv")
	flag := true
	banner("Consulta de Prioridad de Cliente")
	fmt.Println("Ingrese los datos del cliente para determinar su prioridad")
	for flag {
		banner("Tarifa")
		tarifa := read_valid_input(encoderTarifa, "Ingrese Tarifa: ")
		banner("Departamento")
		departamento := read_valid_input(encoderDepartamento, "Ingrese Departamento: ")

		banner("Provincia")
		provincia := read_valid_input(encoderProvincia, "Ingrese Provincia: ")

		banner("Distrito")
		distrito := read_valid_input(encoderDistrito, "Ingrese Distrito: ")

		banner("Cartera")
		cartera := read_valid_input(encoderCartera, "Ingrese Cartera: ")

		fmt.Println("\nDatos ingresados:")
		fmt.Println("Tarifa:", tarifa)
		fmt.Println("Departamento:", departamento)
		fmt.Println("Provincia:", provincia)
		fmt.Println("Distrito:", distrito)
		fmt.Println("Cartera:", cartera)

		transform := []float64{
			encoderTarifa[tarifa], encoderDepartamento[departamento], encoderProvincia[provincia], encoderDistrito[distrito], encoderCartera[cartera]}
		matrix := make([][]float64, 1)
		matrix[0] = transform
		if rfc.Predict(matrix)[0] == 0 {
			println("El cliente no tiene prioridad")
		} else {
			println("El cliente es alta prioridad, se recomienda estar atento para atención y seguimiento")
		}
		fmt.Println("Realizar otra Conulta de otro cliente ?")
		if readInput("Ingrese 's' para continuar: ") == "s" {
			flag = true
		} else {
			flag = false
			fmt.Println("Gracias por usar el sistema")
		}
	}

}

func main() {
	/*
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
	/*	ccRfModel := ccrandomforest.NewRandomForestClassifier(numTrees, maxDepth, maxFeatures, 100)
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
	*/
	app()
}
