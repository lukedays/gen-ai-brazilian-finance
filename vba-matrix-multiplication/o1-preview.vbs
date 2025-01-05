Option Explicit

Sub MultiplyMatrices()
    Dim rngA As Range
    Dim rngB As Range
    Dim rngResult As Range
    Dim A(1 To 3, 1 To 3) As Double
    Dim B(1 To 3, 1 To 3) As Double
    Dim Result(1 To 3, 1 To 3) As Double
    Dim i As Integer, j As Integer, k As Integer

    On Error Resume Next
    ' Prompt user to select first matrix
    Set rngA = Application.InputBox("Select the first 3x3 matrix:", "Matrix A", Type:=8)
    If rngA Is Nothing Then Exit Sub 'User canceled
    If rngA.Cells.Count <> 9 Then
        MsgBox "Please select exactly 3x3 cells for Matrix A."
        Exit Sub
    End If
    
    ' Prompt user to select second matrix
    Set rngB = Application.InputBox("Select the second 3x3 matrix:", "Matrix B", Type:=8)
    If rngB Is Nothing Then Exit Sub 'User canceled
    If rngB.Cells.Count <> 9 Then
        MsgBox "Please select exactly 3x3 cells for Matrix B."
        Exit Sub
    End If
    
    ' Prompt user to select output range
    Set rngResult = Application.InputBox("Select the output 3x3 range for the result:", "Result Matrix", Type:=8)
    If rngResult Is Nothing Then Exit Sub 'User canceled
    If rngResult.Cells.Count <> 9 Then
        MsgBox "Please select exactly 3x3 cells for the result."
        Exit Sub
    End If
    On Error GoTo 0
    
    ' Read matrices A and B
    For i = 1 To 3
        For j = 1 To 3
            A(i, j) = rngA.Cells(i, j).Value
            B(i, j) = rngB.Cells(i, j).Value
        Next j
    Next i
    
    ' Multiply matrices
    For i = 1 To 3
        For j = 1 To 3
            Result(i, j) = 0
            For k = 1 To 3
                Result(i, j) = Result(i, j) + A(i, k) * B(k, j)
            Next k
        Next j
    Next i
    
    ' Write result to output range
    For i = 1 To 3
        For j = 1 To 3
            rngResult.Cells(i, j).Value = Result(i, j)
        Next j
    Next i
    
    MsgBox "Matrix multiplication completed successfully."
    
End Sub