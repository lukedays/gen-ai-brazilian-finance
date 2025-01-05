Sub Multiply3x3Matrices()
    Dim MatrixA(1 To 3, 1 To 3) As Double
    Dim MatrixB(1 To 3, 1 To 3) As Double
    Dim Result(1 To 3, 1 To 3) As Double
    Dim i As Integer, j As Integer, k As Integer
    Dim ws As Worksheet
    
    ' Set the worksheet (modify if needed)
    Set ws = ThisWorkbook.ActiveSheet
    
    ' Read Matrix A from A1:C3
    For i = 1 To 3
        For j = 1 To 3
            MatrixA(i, j) = ws.Cells(i, j).Value
        Next j
    Next i
    
    ' Read Matrix B from E1:G3
    For i = 1 To 3
        For j = 1 To 3
            MatrixB(i, j) = ws.Cells(i, j + 4).Value ' E is column 5
        Next j
    Next i
    
    ' Perform Matrix Multiplication: Result = MatrixA * MatrixB
    For i = 1 To 3
        For j = 1 To 3
            Result(i, j) = 0
            For k = 1 To 3
                Result(i, j) = Result(i, j) + MatrixA(i, k) * MatrixB(k, j)
            Next k
        Next j
    Next i
    
    ' Output the Result to I1:K3
    For i = 1 To 3
        For j = 1 To 3
            ws.Cells(i, j + 8).Value = Result(i, j) ' I is column 9
        Next j
    Next i
    
    MsgBox "Matrix multiplication complete! Result is in cells I1:K3.", vbInformation
End Sub