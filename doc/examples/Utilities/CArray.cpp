#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("CArray");

//! [CArray]
#include "TBTK/CArray.h"
#include "TBTK/Streams.h"

using namespace TBTK;

int main(){
	CArray<unsigned int> carray(10);
	for(unsigned int n = 0; n < carray.getSize(); n++)
		carray[n] = n;

	for(unsigned int n = 0; n < carray.getSize(); n++)
		Streams::out << carray[n] << "\t";
	Streams::out << "\n";

	unsigned int *rawArray = carray.getData();
	for(unsigned int n = 0; n < 10; n++)
		Streams::out << rawArray[n] << "\t";
	Streams::out << "\n";
}
//! [CArray]
