#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("Index");

//! [Index]
#include "TBTK/Index.h"
#include "TBTK/Streams.h"

using namespace TBTK;

int main(){
	Index index({1, 2, 3});
	for(unsigned int n = 0; n < index.getSize(); n++)
		Streams::out << index[n] << "\n";
}
//! [Index]
