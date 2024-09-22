package test

import "fmt"

// CalcularPrecision calcula la precision
func CalcPrecision(y, predictY []float64) float64 {
	var truePositive, falsePositive float64
	for i := 0; i < len(y); i++ {
		if predictY[i] == 1 && y[i] == 1 {
			truePositive++
		}
		if predictY[i] == 1 && y[i] == 0 {
			falsePositive++
		}
	}
	if truePositive+falsePositive == 0 {
		return 0
	}
	return truePositive / (truePositive + falsePositive)
}

// CalcularRecall calcula el recall
func CalcRecall(y, predictY []float64) float64 {
	var truePositive, falseNegative float64
	for i := 0; i < len(y); i++ {
		if predictY[i] == 1 && y[i] == 1 {
			truePositive++
		}
		if predictY[i] == 0 && y[i] == 1 {
			falseNegative++
		}
	}
	if truePositive+falseNegative == 0 {
		return 0
	}
	return truePositive / (truePositive + falseNegative)
}

// CalcularF1Score calcula el F1 score
func CalcF1Score(y, predictY []float64) float64 {
	precision := CalcPrecision(y, predictY)
	recall := CalcRecall(y, predictY)
	if precision+recall == 0 {
		return 0
	}
	return 2 * (precision * recall) / (precision + recall)
}

/*
func main() {
	// Ejemplo de uso
	y := []float64{1, 0, 1, 1, 0, 1, 0, 0, 1, 0}        // Etiquetas verdaderas
	predictY := []float64{1, 0, 1, 0, 0, 1, 0, 1, 1, 0} // Predicciones

	precision := CalcularPrecision(y, predictY)
	recall := CalcularRecall(y, predictY)
	f1Score := CalcularF1Score(y, predictY)

	fmt.Printf("Precision: %.2f\n", precision)
	fmt.Printf("Recall: %.2f\n", recall)
	fmt.Printf("F1 Score: %.2f\n", f1Score)
}
*/
func ConfusionMatrix(y_true, y_pred []float64) {
	tp, fp, tn, fn := 0, 0, 0, 0

	for i := range y_true {
		if y_true[i] == 1 && y_pred[i] == 1 {
			tp++
		} else if y_true[i] == 0 && y_pred[i] == 1 {
			fp++
		} else if y_true[i] == 0 && y_pred[i] == 0 {
			tn++
		} else if y_true[i] == 1 && y_pred[i] == 0 {
			fn++
		}
	}

	fmt.Println("Matriz de Confusión:")
	fmt.Printf("                Predict Positive   Predict Negative\n")
	fmt.Printf("Real Positive   %20d   %20d\n", tp, fn)
	fmt.Printf("Real Negative   %20d   %20d\n", fp, tn)
}
