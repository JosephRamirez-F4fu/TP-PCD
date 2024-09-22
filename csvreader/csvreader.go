package csvreader

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
)

func ReadCSV(filePath string) ([]string, [][]string, error) {
	// Abrir el archivo CSV
	file, err := os.Open(filePath)
	if err != nil {
		return nil, nil, fmt.Errorf("Error. No se puede abrir el archivo: %v", err)
	}
	defer file.Close()

	// Crear un lector CSV
	reader := csv.NewReader(file)

	// Leer todas las filas del archivo CSV
	rows, err := reader.ReadAll()
	if err != nil {
		return nil, nil, fmt.Errorf("Error. Al leer el archivo CSV: %v", err)
	}

	// Verificar si hay filas
	if len(rows) == 0 {
		return nil, nil, fmt.Errorf("El archivo CSV está vacío")
	}

	return rows[0], rows[1:], nil
}

/*
// Función para leer un archivo CSV y devolver una matriz de strings (incluyendo la cabecera)
func ReadCSV(filePath string) ([]string, [][]string, error) {
	// Abrir el archivo CSV
	file, err := os.Open(filePath)
	if err != nil {
		return nil, nil, fmt.Errorf("Error. No se puede abrir el archivo: %v", err)
	}
	defer file.Close()

	// Crear un lector CSV
	reader := csv.NewReader(file)

	// Leer todas las filas del archivo CSV
	rows, err := reader.ReadAll()
	if err != nil {
		return nil, nil, fmt.Errorf("Error. Al leer el archivo CSV: %v", err)
	}

	return rows[0], rows[1:], nil
}*/

// Función para convertir una matriz de strings a float64, omitiendo la primera fila
func ConvertToFloat64(data [][]string) ([][]float64, error) {
	// Crear la matriz para almacenar los datos convertidos
	floatData := make([][]float64, len(data))
	for i, row := range data {
		floatData[i] = make([]float64, len(row))
		for j, cell := range row {
			value, err := strconv.ParseFloat(cell, 64)
			if err != nil {
				return nil, fmt.Errorf("error al convertir el valor '%s' a float64: %v", cell, err)
			}
			floatData[i][j] = value
		}
	}
	return floatData, nil
}

/*
func main() {
	// Ejemplo de uso del archivo CSV
	filePath := "train.csv"

	// Leer el archivo CSV como una matriz de strings
	columns, textData, err := readCSV(filePath)
	fmt.Printf("Columns: %v", columns)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	numData := make([][]string, len(textData))
	for i, row := range textData {
		numData[i] = row[1:]
	}

	// Convertir los datos a una matriz de float64
	floatData, err := convertToFloat64(numData)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	X, y := make([][]float64, len(textData)), make([]float64, len(textData))
	for i, row := range floatData {
		X[i] = row[1:]
		y[i] = row[0]
	}
	println("\nX rows=", len(X), "X cols:", len(X[0]))
	println("\ny rows=", len(y))

	// Imprimir la matriz de datos convertidos
	//for _, row := range y {
	//	fmt.Println(row)
	//}
}
*/
