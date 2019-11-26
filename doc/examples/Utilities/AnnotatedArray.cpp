#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("AnnotatedArray");

//! [AnnotatedArray]
#include "TBTK/AnnotatedArray.h"
#include "TBTK/Streams.h"

#include <vector>

using namespace std;
using namespace TBTK;

int main(){
	//Create Arrays and axes.
	Array<unsigned int> array({2, 3});
	std::vector<std::vector<double>> axes = {
		{0, 1},
		{1, 1.5, 2}
	};

	//Fill the Array with values.
	for(unsigned int x = 0; x < 2; x++){
		for(unsigned int y = 0; y < 3; y++){
			array[{x, y}] = x*y;
		}
	}

	//Bundle the array and axes and an AnnotatedArray.
	AnnotatedArray<unsigned int, double> annotatedArray(array, axes);

	//Print the Array.
	Streams::out << annotatedArray << "\n";

	//Get and print the axes.
	const std::vector<std::vector<double>> &annotatedAxes
		= annotatedArray.getAxes();
	Streams::out << "Axes:\n";
	for(unsigned int n = 0; n < annotatedAxes.size(); n++){
		Streams::out << "\t" << Array<double>(annotatedAxes[n])
			<< "\n";
	}
}
//! [AnnotatedArray]
