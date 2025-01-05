Sub MatrixMultiplication()
    ' Define ranges for input matrices and output
    Dim matrix1 As Range
    Dim matrix2 As Range
    Dim resultMatrix As Range
    
    ' Set input ranges (A1:C3 for first matrix, E1:G3 for second matrix)
    Set matrix1 = Range("A1:C3")
    Set matrix2 = Range("E1:G3")
    ' Set output range (I1:K3)
    Set resultMatrix = Range("I1:K3")
    
    ' Variables for calculation
    Dim i As Integer, j As Integer, k As Integer
    Dim sum As Double
    
    ' Perform matrix multiplication
    For i = 1 To 3
        For j = 1 To 3
            sum = 0
            For k = 1 To 3
                sum = sum + matrix1.Cells(i, k).Value * matrix2.Cells(k, j).Value
            Next k
            resultMatrix.Cells(i, j).Value = sum
        Next j
    Next i
    
    ' Format result cells
    resultMatrix.NumberFormat = "0.00"
    resultMatrix.Font.Bold = True
End Sub