#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("Range");

//! [Range]
#include "TBTK/Range.h"
#include "TBTK/Streams.h"

using namespace std;
using namespace TBTK;

void print(const Range &range){
	for(unsigned int n = 0; n < range.getResolution(); n++){
		if(n != 0)
			Streams::out << ", ";
		Streams::out << range[n];
	}
}

int main(){
	Streams::out << "Range [-1, 1]: ";
	Range range(-1, 1, 5);
	print(range);

	Streams::out << "\nRange (-1, 1]: ";
	range = Range(-1, 1, 5, false, true);
	print(range);

	Streams::out << "\nRange [-1, 1): ";
	range = Range(-1, 1, 5, true, false);
	print(range);

	Streams::out << "\nRange (-1, 1): ";
	range = Range(-1, 1, 5, false, false);
	print(range);

	Streams::out << "\n";
}
//! [Range]
