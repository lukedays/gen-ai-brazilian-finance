Sub MatrixMultiplication()
    Dim MatrixA(1 To 3, 1 To 3) As Double
    Dim MatrixB(1 To 3, 1 To 3) As Double
    Dim ResultMatrix(1 To 3, 1 To 3) As Double
    Dim i As Integer, j As Integer, k As Integer
    
    ' Read Matrix A from cells A1:C3
    For i = 1 To 3
        For j = 1 To 3
            MatrixA(i, j) = Cells(i, j).Value
        Next j
    Next i
    
    ' Read Matrix B from cells E1:G3
    For i = 1 To 3
        For j = 1 To 3
            MatrixB(i, j) = Cells(i, j + 4).Value
        Next j
    Next i
    
    ' Perform Matrix Multiplication
    For i = 1 To 3
        For j = 1 To 3
            ResultMatrix(i, j) = 0
            For k = 1 To 3
                ResultMatrix(i, j) = ResultMatrix(i, j) + MatrixA(i, k) * MatrixB(k, j)
            Next k
        Next j
    Next i
    
    ' Output the Result Matrix to cells A5:C7
    For i = 1 To 3
        For j = 1 To 3
            Cells(i + 4, j).Value = ResultMatrix(i, j)
        Next j
    Next i
End Sub