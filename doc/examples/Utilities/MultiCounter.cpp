#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("MultiCounter");

//! [MultiCounter]
#include "TBTK/MultiCounter.h"
#include "TBTK/Streams.h"

using namespace std;
using namespace TBTK;

int main(){
	Streams::out << "Three level nested loop:\n";
	for(unsigned int x = 0; x < 3; x++){
		for(unsigned int y = 1; y < 7; y += 2){
			for(unsigned int z = 2; z < 8; z += 3){
				Streams::out << x << "\t";
				Streams::out << y << "\t";
				Streams::out << z << "\n";
			}
		}
	}

	Streams::out << "\nEquivalent flattened loop:\n";
	MultiCounter<unsigned int> multiCounter(
		{0, 1, 2},
		{3, 7, 8},
		{1, 2, 3}
	);
	for(multiCounter.reset(); !multiCounter.done(); ++multiCounter){
		unsigned int x = multiCounter[0];
		unsigned int y = multiCounter[1];
		unsigned int z = multiCounter[2];
		Streams::out << x << "\t";
		Streams::out << y << "\t";
		Streams::out << z << "\n";
	}
}
//! [MultiCounter]
