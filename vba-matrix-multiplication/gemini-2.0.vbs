Sub MatrixMultiply()

  ' Declare variables
  Dim matrixA(1 To 3, 1 To 3) As Double
  Dim matrixB(1 To 3, 1 To 3) As Double
  Dim resultMatrix(1 To 3, 1 To 3) As Double
  Dim i As Integer, j As Integer, k As Integer

  ' Get input from cells (Assuming Matrix A is in A1:C3 and Matrix B is in E1:G3)
  For i = 1 To 3
    For j = 1 To 3
      matrixA(i, j) = ThisWorkbook.Sheets("Sheet1").Cells(i, j).Value ' Change "Sheet1" if needed
      matrixB(i, j) = ThisWorkbook.Sheets("Sheet1").Cells(i, j + 4).Value ' Offset by 4 columns for Matrix B
    Next j
  Next i

  ' Perform matrix multiplication
  For i = 1 To 3 ' Rows of result
    For j = 1 To 3 ' Columns of result
      resultMatrix(i, j) = 0
      For k = 1 To 3 ' Inner loop for multiplication
        resultMatrix(i, j) = resultMatrix(i, j) + (matrixA(i, k) * matrixB(k, j))
      Next k
    Next j
  Next i

  ' Output the result to cells (Starting at I1)
  For i = 1 To 3
    For j = 1 To 3
      ThisWorkbook.Sheets("Sheet1").Cells(i, j + 8).Value = resultMatrix(i, j) 'Offset by 8 columns for output
    Next j
  Next i

  MsgBox "Matrix multiplication complete. Result is in I1:K3", vbInformation

End Sub