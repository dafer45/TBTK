#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("HoppingAmplitude");

//! [HoppingAmplitude]
#include "TBTK/HoppingAmplitude.h"
#include "TBTK/Streams.h"

#include <complex>

using namespace std;
using namespace TBTK;

int main(){
	HoppingAmplitude hoppingAmplitude(1, {1, 2, 3}, {4, 5});
	Streams::out << hoppingAmplitude << "\n";

	std::complex<double> amplitude = hoppingAmplitude.getAmplitude();
	Index toIndex = hoppingAmplitude.getToIndex();
	Index fromIndex = hoppingAmplitude.getFromIndex();

	Streams::out << amplitude << "\n";
	Streams::out << toIndex.toString() << "\n";
	Streams::out << fromIndex.toString() << "\n";
}
//! [HoppingAmplitude]
