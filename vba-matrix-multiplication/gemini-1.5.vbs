Sub MatrixMultiplication()

  ' Define the ranges for the two matrices
  Dim rngMatrix1 As Range
  Dim rngMatrix2 As Range

  ' Set the ranges (adjust these to match your actual data)
  Set rngMatrix1 = Range("A1:C3")  ' Matrix 1
  Set rngMatrix2 = Range("E1:G3")  ' Matrix 2

  ' Get the matrix values
  Dim mat1() As Variant
  Dim mat2() As Variant
  mat1 = rngMatrix1.Value
  mat2 = rngMatrix2.Value

  ' Define the output range
  Dim rngOutput As Range
  Set rngOutput = Range("I1")  ' Top-left cell of the output matrix

  ' Perform matrix multiplication
  Dim result() As Variant
  result = MatrixMultiply(mat1, mat2)

  ' Write the result to the output range
  rngOutput.Resize(UBound(result, 1), UBound(result, 2)).Value = result

End Sub

Function MatrixMultiply(ByRef mat1 As Variant, ByRef mat2 As Variant) As Variant

  ' Check for compatible dimensions
  If UBound(mat1, 2) <> UBound(mat2, 1) Then
    MatrixMultiply = CVErr(xlErrValue)  ' Return an error if dimensions are incompatible
    Exit Function
  End If

  ' Create an array for the result
  Dim result() As Variant
  ReDim result(1 To UBound(mat1, 1), 1 To UBound(mat2, 2))

  ' Perform the multiplication
  Dim i As Long, j As Long, k As Long
  For i = 1 To UBound(result, 1)
    For j = 1 To UBound(result, 2)
      result(i, j) = 0
      For k = 1 To UBound(mat1, 2)
        result(i, j) = result(i, j) + mat1(i, k) * mat2(k, j)
      Next k
    Next j
  Next i

  ' Return the result
  MatrixMultiply = result

End Function