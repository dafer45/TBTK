/** @package TBTKcalc
 *  @file Gauge.h
 *  @brief Generalized gauge transformation.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_GAUGE
#ifndef COM_DAFER45_TBTK_GAUGE

#include <complex>

namespace TBTK{

class Gauge{
public:
	/** Constructor. */
	Gauge();

	/** Destructor. */
	~Gauge();
private:
	/** Number of HoppingAmplitudes before the Gauge transformation. */
	int numOriginalHoppingAmplitudes;

	/** Signs to apply to coefficients. */
	int *signs;

	/** Flags indicating whether or not to apply complex conjugation to coefficients. */
	bool *conjugations;

	/** Column indices for transformations. */
	int *unitaryColIndices;

	/** Row indices for transformation transformation. */
	int *unitaryRowIndices;

	/** Values of the unitary transformation. */
	complex<double> *unitaryValues;
};

};	//End of namespace TBTK

#endif
