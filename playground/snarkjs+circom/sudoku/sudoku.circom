pragma circom 2.0.0;

template NonEqual(){
    signal input in0;
    signal input in1;
    signal inv;
    inv <-- 1/ (in0 - in1);
    inv*(in0 - in1) === 1;
}

template Distinct(n) {
    signal input in[n];
    component nonEqual[n][n];
    for(var i = 0; i < n; i++){
        for(var j = 0; j < i; j++){
            nonEqual[i][j] = NonEqual();
            nonEqual[i][j].in0 <== in[i];
            nonEqual[i][j].in1 <== in[j];
        }
    }
}

// Enforce that 0 <= in < 16
template Bits4(){
    signal input in;
    signal bits[4];
    var bitsum = 0;
    for (var i = 0; i < 4; i++) {
        bits[i] <-- (in >> i) & 1;
        bits[i] * (bits[i] - 1) === 0;
        bitsum = bitsum + 2 ** i * bits[i];
    }
    bitsum === in;
}

// Enforce that 1 <= in <= 9
template OneToNine() {
    signal input in;
    component lowerBound = Bits4();
    component upperBound = Bits4();
    lowerBound.in <== in - 1;
    upperBound.in <== in + 6;
}

// Template for an (n^2) by (n^2) sudoku composed of an n-by-n grid of n-by-n boxes
//
// n being the square root of the size instead of the actual size is necessary to avoid
// having to evaluate the size of the boxes in the circuit compilation
template Sudoku(n) {
    // solution is a 2D array: indices are (row, col)
    signal input solution[n**2][n**2];
    // puzzle is the same, but a zero indicates a blank
    signal input puzzle[n**2][n**2];

    component distinctRows[n**2];
    component distinctColumns[n**2];
    component distinctBoxes[n**2];
    component inRange[n**2][n**2];

	// Check that solution matches initial puzzle (where defined)
    for (var row = 0; row < n**2; row++) {
        for (var col = 0; col < n**2; col++) {
            puzzle[row][col] * (puzzle[row][col] - solution[row][col]) === 0;
        }
    }

    // Check that solution is in range
    for (var row = 0; row < n**2; row++) {
        for (var col = 0; col < n**2; col++) {
            inRange[row][col] = OneToNine();
            inRange[row][col].in <== solution[row][col];
        }
    }

    // Check that rows are distinct
    for (var row = 0; row < n**2; row++) {
        distinctRows[row] = Distinct(n**2);
        for (var col = 0; col < n**2; col++) {
            distinctRows[row].in[col] <== solution[row][col];
        }
    }

    // Check that columns are distinct
    for (var col = 0; col < n**2; col++) {
        distinctColumns[col] = Distinct(n**2);
        for (var row = 0; row < n**2; row++) {
            distinctColumns[col].in[row] <== solution[row][col];
        }
    }

    // Check that boxes are distinct
    for (var box_row = 0; box_row < n; box_row++) {
		for (var box_col = 0; box_col < n; box_col++) {
			var box = box_row + box_col * n;
			distinctBoxes[box] = Distinct(n**2);

			for (var cell_row = 0; cell_row < n; cell_row++) {
				for(var cell_col = 0; cell_col < n; cell_col++) {
					var index = cell_row + cell_col * n;
					var global_row = cell_row + box_row * n;
					var global_col = cell_col + box_col * n;
					distinctBoxes[box].in[index] <== solution[global_row][global_col];
				}
			}
		}
    }
}

component main {public[puzzle]} = Sudoku(3);
