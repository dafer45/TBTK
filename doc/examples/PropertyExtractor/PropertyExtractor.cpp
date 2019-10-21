#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("PropertyExtractor");

//! [PropertyExtractor]
#include "TBTK/PropertyExtractor/PropertyExtractor.h"

using namespace TBTK;

int main(){
	PropertyExtractor::PropertyExtractor propertyExtractor;

	//Set the energy window for which to extract energy dependent
	//Properties.
	double lowerBound = -10;
	double upperBound = -10;
	double resolution = 1000;
	propertyExtractor.setEnergyWindow(lowerBound, upperBound, resolution);

	//Set the size of the imaginary infinitesimal to use in the denominator
	//of the Green's function if it is used during the calculation.
	propertyExtractor.setEnergyInfinitesimal(1e-10);
}
//! [PropertyExtractor]
