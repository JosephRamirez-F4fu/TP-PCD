package sqrandomforest

import (
	"math/rand"
	"sort"
)

// Estructura que define un árbol de decisión.
type DecisionTree struct {
	IsLeaf       bool
	Prediction   float64
	FeatureIndex int
	Threshold    float64
	Left         *DecisionTree
	Right        *DecisionTree
}

// Estructura que define el Random Forest.
type RandomForest struct {
	Trees       []*DecisionTree
	NumTrees    int
	MaxDepth    int
	MaxFeatures int
	r           *rand.Rand
}

func NewRandomForestClassifier(numTrees int, maxDepth int, maxFeatures int, randomState int) RandomForest {
	source := rand.NewSource(int64(randomState))
	r := rand.New(source)
	rfc := RandomForest{
		NumTrees:    numTrees,
		MaxDepth:    maxDepth,
		MaxFeatures: maxFeatures,
		r:           r,
	}
	return rfc
}

// Función para entrenar un Random Forest.
func (model *RandomForest) Train(X [][]float64, y []float64) {
	model.Trees = make([]*DecisionTree, model.NumTrees)

	for i := 0; i < model.NumTrees; i++ {
		// Crear bootstrap sample
		sampledX, sampledY := bootstrapSample(X, y, model.r)
		// Seleccionar características aleatorias
		selectedFeatures := selectRandomFeatures(len(X[0]), model.MaxFeatures, model.r)
		// Filtrar datos solo con las características seleccionadas
		reducedX := filterXByFeatures(sampledX, selectedFeatures)
		// Entrenar el árbol de decisión
		model.Trees[i] = trainDecisionTree(reducedX, sampledY, 0, model.MaxDepth, selectedFeatures)
	}
}

// Función para predecir la clase para un nuevo dato.
func (model *RandomForest) predict(X []float64) float64 {
	votes := make(map[float64]int)
	for _, tree := range model.Trees {
		vote := predictFromTree(tree, X)
		votes[vote]++
	}

	// Devolver la clase con más votos
	var majorityClass float64
	maxVotes := 0
	for class, count := range votes {
		if count > maxVotes {
			maxVotes = count
			majorityClass = class
		}
	}
	return majorityClass
}

func (model *RandomForest) Predict(X [][]float64) []float64 {
	predY := make([]float64, len(X))
	for i, row := range X {
		predY[i] = model.predict(row)
	}
	return predY
}

// Función para realizar una predicción utilizando un árbol de decisión.
func predictFromTree(tree *DecisionTree, X []float64) float64 {
	if tree.IsLeaf {
		return tree.Prediction
	}

	if X[tree.FeatureIndex] <= tree.Threshold {
		return predictFromTree(tree.Left, X)
	}
	return predictFromTree(tree.Right, X)
}

// Función para entrenar un árbol de decisión con las características seleccionadas.
func trainDecisionTree(X [][]float64, y []float64, depth, maxDepth int, selectedFeatures []int) *DecisionTree {
	if depth == maxDepth || isPure(y) {
		leaf := &DecisionTree{IsLeaf: true, Prediction: majorityClass(y)}
		return leaf
	}

	// Encontrar la mejor característica y umbral solo entre las características seleccionadas
	featureIndex, threshold := findBestSplit(X, y)

	if featureIndex == -1 {
		return &DecisionTree{IsLeaf: true, Prediction: majorityClass(y)}
	}
	leftIndices, rightIndices := splitX(X, featureIndex, threshold)

	if len(leftIndices) == 0 || len(rightIndices) == 0 {
		return &DecisionTree{IsLeaf: true, Prediction: majorityClass(y)}
	}

	leftX, leftY := filterX(X, y, leftIndices)
	rightX, rightY := filterX(X, y, rightIndices)

	return &DecisionTree{
		FeatureIndex: selectedFeatures[featureIndex], // Usar el índice de la característica original
		Threshold:    threshold,
		Left:         trainDecisionTree(leftX, leftY, depth+1, maxDepth, selectedFeatures),
		Right:        trainDecisionTree(rightX, rightY, depth+1, maxDepth, selectedFeatures),
	}
}

// Función para seleccionar un subconjunto aleatorio de características.
func selectRandomFeatures(numFeatures, maxFeatures int, r *rand.Rand) []int {
	selectedFeatures := r.Perm(numFeatures)[:maxFeatures]
	return selectedFeatures
}

// Función para filtrar los datos solo con las características seleccionadas.
func filterXByFeatures(X [][]float64, selectedFeatures []int) [][]float64 {
	reducedX := make([][]float64, len(X))
	for i, row := range X {
		newRow := make([]float64, len(selectedFeatures))
		for j, featureIndex := range selectedFeatures {
			newRow[j] = row[featureIndex]
		}
		reducedX[i] = newRow
	}
	return reducedX
}

// Función para calcular la impureza de Gini para un conjunto de etiquetas.
func giniImpurity(y []float64) float64 {
	labelCounts := make(map[float64]int)
	for _, label := range y {
		labelCounts[label]++
	}

	totalY := float64(len(y))
	gini := 1.0
	for _, count := range labelCounts {
		probability := float64(count) / totalY
		gini -= probability * probability
	}
	return gini
}

// Función para calcular la mediana de una característica.
func median(values []float64) float64 {
	n := len(values)
	sortedValues := make([]float64, n)
	copy(sortedValues, values)
	sort.Float64s(sortedValues)

	if n%2 == 0 {
		return (sortedValues[n/2-1] + sortedValues[n/2]) / 2.0
	}
	return sortedValues[n/2]
}

// Función para encontrar la mejor característica y umbral (mediana) para dividir los datos usando la impureza de Gini.
func findBestSplit(X [][]float64, y []float64) (int, float64) {
	numFeatures := len(X[0])
	bestFeatureIndex := -1
	bestThreshold := 0.0
	bestGini := 1.0

	for featureIndex := 0; featureIndex < numFeatures; featureIndex++ {
		// Extraer la característica actual
		featureValues := make([]float64, len(X))
		for i := range X {
			featureValues[i] = X[i][featureIndex]
		}

		// Calcular la mediana como umbral
		threshold := median(featureValues)

		// Dividir los datos según la mediana
		leftIndices, rightIndices := splitX(X, featureIndex, threshold)

		// Si no hay datos en alguna de las ramas, saltar este umbral
		if len(leftIndices) == 0 || len(rightIndices) == 0 {
			continue
		}

		// Obtener las etiquetas de los conjuntos izquierdo y derecho
		leftY := []float64{}
		rightY := []float64{}
		for _, idx := range leftIndices {
			leftY = append(leftY, y[idx])
		}
		for _, idx := range rightIndices {
			rightY = append(rightY, y[idx])
		}

		// Calcular la impureza Gini ponderada para la división
		leftGini := giniImpurity(leftY)
		rightGini := giniImpurity(rightY)
		weightedGini := (float64(len(leftY))*leftGini + float64(len(rightY))*rightGini) / float64(len(y))

		// Si la impureza Gini es mejor (menor), actualizar el mejor umbral y característica
		if weightedGini < bestGini {
			bestGini = weightedGini
			bestFeatureIndex = featureIndex
			bestThreshold = threshold
		}
	}

	return bestFeatureIndex, bestThreshold
}

// Función para dividir los datos según el umbral de una característica.
func splitX(X [][]float64, featureIndex int, threshold float64) ([]int, []int) {
	leftIndices := []int{}
	rightIndices := []int{}
	for i, row := range X {
		if row[featureIndex] <= threshold {
			leftIndices = append(leftIndices, i)
		} else {
			rightIndices = append(rightIndices, i)
		}
	}
	return leftIndices, rightIndices
}

// Función para filtrar los datos según los índices.
func filterX(X [][]float64, y []float64, indices []int) ([][]float64, []float64) {
	filteredX := [][]float64{}
	filteredY := []float64{}
	for _, idx := range indices {
		filteredX = append(filteredX, X[idx])
		filteredY = append(filteredY, y[idx])
	}
	return filteredX, filteredY
}

// Función para verificar si las etiquetas son puras (todas iguales).
func isPure(y []float64) bool {
	firstLabel := y[0]
	for _, label := range y {
		if label != firstLabel {
			return false
		}
	}
	return true
}

// Función para encontrar la clase mayoritaria en un conjunto de etiquetas.
func majorityClass(y []float64) float64 {
	classCounts := make(map[float64]int)
	for _, label := range y {
		classCounts[label]++
	}

	var majorityClass float64
	maxCount := 0
	for class, count := range classCounts {
		if count > maxCount {
			maxCount = count
			majorityClass = class
		}
	}
	return majorityClass
}

// Función para crear un bootstrap sample de los datos.
func bootstrapSample(X [][]float64, y []float64, r *rand.Rand) ([][]float64, []float64) {
	n := len(X)
	sampledX := make([][]float64, n)
	sampledY := make([]float64, n)
	for i := 0; i < n; i++ {
		index := r.Intn(n)
		sampledX[i] = X[index]
		sampledY[i] = y[index]
	}
	return sampledX, sampledY
}

/*
// Ejemplo de uso
func main() {
	// Datos de ejemplo
	X := [][]float64{
		{2.7, 2.5},
		{1.4, 2.3},
		{3.3, 4.4},
		{1.3, 1.8},
		{3.0, 3.0},
	}
	y := []float64{0, 0, 1, 0, 1}

	// Crear y entrenar el Random Forest
	model := RandomForest{NumTrees: 10, MaxDepth: 3, MaxFeatures: 2}
	model.Train(X, y)

	// Hacer predicciones
	testX := []float64{2.7, 2.5}
	prediction := model.Predict(testX)

	fmt.Printf("Predicción: %.2f\n", prediction)

}
*/
