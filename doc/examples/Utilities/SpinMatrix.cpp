#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("SpinMatrix");

//! [SpinMatrix]
#include "TBTK/SpinMatrix.h"
#include "TBTK/Streams.h"

#include <complex>

using namespace std;
using namespace TBTK;

int main(){
	complex<double> i(0, 1);
	double rho = 1;
	double S_x = 0.1;
	double S_y = 0.2;
	double S_z = 0.3;
	SpinMatrix spinMatrix;
	spinMatrix.at(0, 0) = (rho + S_z)/2.;
	spinMatrix.at(1, 0) = (S_x - i*S_y)/2.;
	spinMatrix.at(0, 1) = (S_x + i*S_y)/2.;
	spinMatrix.at(1, 1) = (rho - S_z)/2.;

	Streams::out << spinMatrix << "\n";
	Streams::out << "Density: " << spinMatrix.getDensity() << "\n";
	Streams::out << "Spin vector: " << spinMatrix.getSpinVector() << "\n";
}
//! [SpinMatrix]
