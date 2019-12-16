#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("Array");

//! [Array]
#include "TBTK/Array.h"
#include "TBTK/Streams.h"
#include "TBTK/TBTK.h"

#include <vector>

using namespace std;
using namespace TBTK;

int main(){
	Initialize();

	//Create Arrays.
	Array<unsigned int> array0({2, 3, 4});
	Array<unsigned int> array1({2, 3, 4});
	Array<unsigned int> array2({2, 3, 4});

	//Fill Arrays with values.
	for(unsigned int x = 0; x < 2; x++){
		for(unsigned int y = 0; y < 3; y++){
			for(unsigned int z = 0; z < 4; z++){
				array0[{x, y, z}] = x;
				array1[{x, y, z}] = 2*y;
				array2[{x, y, z}] = 3*z;
			}
		}
	}

	//Perform arithmetic operations on Arrays.
	Array<unsigned int> result = array0 - array1/2 + 3*array2;

	//Get the ranges for the result.
	const vector<unsigned int> &ranges = result.getRanges();
	Streams::out << "Result dimension: " << ranges.size() << "\n";
	Streams::out << "Result ranges: ";
	for(unsigned int n = 0; n < ranges.size(); n++)
		Streams::out << ranges[n] << "\t";
	Streams::out << "\n";

	//Get slice containing data for all indices of the form {_a_, 0, _a_},
	//where _a_ is a wildcard.
	Array<unsigned int> slice = result.getSlice({_a_, 0, _a_});

	//Get the ranges for the slice.
	const vector<unsigned int> &sliceRanges = slice.getRanges();
	Streams::out << "Slice dimension: " << sliceRanges.size() << "\n";
	Streams::out << "Slice ranges: ";
	for(unsigned int n = 0; n < sliceRanges.size(); n++)
		Streams::out << sliceRanges[n] << "\t";
	Streams::out << "\n";

	//Print the values in the slice.
	Streams::out << "Slice values:\n";
	for(unsigned int x = 0; x < 2; x++){
		for(unsigned int z = 0; z < 4; z++)
			Streams::out << "\t" << slice[{x, z}];
		Streams::out << "\n";
	}
}
//! [Array]
