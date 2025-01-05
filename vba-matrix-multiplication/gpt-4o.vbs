Sub MatrixMultiplication()
    Dim matrixA(1 To 3, 1 To 3) As Double
    Dim matrixB(1 To 3, 1 To 3) As Double
    Dim resultMatrix(1 To 3, 1 To 3) As Double
    Dim i As Integer, j As Integer, k As Integer

    ' Load matrix A from cells A1:C3
    For i = 1 To 3
        For j = 1 To 3
            matrixA(i, j) = Cells(i, j).Value
        Next j
    Next i

    ' Load matrix B from cells E1:G3
    For i = 1 To 3
        For j = 1 To 3
            matrixB(i, j) = Cells(i, j + 4).Value
        Next j
    Next i

    ' Perform matrix multiplication
    For i = 1 To 3
        For j = 1 To 3
            resultMatrix(i, j) = 0
            For k = 1 To 3
                resultMatrix(i, j) = resultMatrix(i, j) + matrixA(i, k) * matrixB(k, j)
            Next k
        Next j
    Next i

    ' Output result matrix to cells I1:K3
    For i = 1 To 3
        For j = 1 To 3
            Cells(i, j + 8).Value = resultMatrix(i, j)
        Next j
    Next i
End Sub
