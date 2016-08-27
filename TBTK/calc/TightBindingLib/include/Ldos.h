/** @package TBTKcalc
 *  @file Ldos.h
 *  @brief Property container for local density of states (LDOS)
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_LDOS
#define COM_DAFER45_TBTK_LDOS

namespace TBTK{
	class CPropertyExtractor;
	class DPropertyExtractor;
	class FileReader;
namespace Property{

/** Container for local density of states (LDOS). */
class Ldos{
public:
	/** Constructor. */
	Ldos(int dimensions, const int *ranges, double lowerBound, double upperBound, int resolution);

	/** Destructor. */
	~Ldos();

	/** Get the dimension of the LDOS. (Excluding energy dimension). */
	int getDimensions() const;

	/** Get the ranges for the dimensions of the LDOS. */
	const int* getRanges() const;

	/** Get lower bound for the energy. */
	double getLowerBound() const;

	/** Get upper bound for the energy. */
	double getUpperBound() const;

	/** Get energy resolution. (Number of energy intervals) */
	int getResolution() const;

	/** Get the number of data elements. */
	int getSize() const;

	/** Get LDOS data. */
	const double* getData() const;
private:
	/** Dimension of the LDOS. */
	int dimensions;

	/** Ranges for the dimensions of the LDOS. */
	int *ranges;

	/** Lower bound for the energy. */
	double lowerBound;

	/** Upper bound for the energy. */
	double upperBound;

	/** Energy resolution. (Number of energy intervals). */
	int resolution;

	/** Number of data elements. */
	int size;

	/** Actual data. */
	double *data;

	/** CPropertyExtractor is a friend class to allow it to write LDOS
	 *  data. */
	friend class TBTK::CPropertyExtractor;

	/** DPropertyExtractor is a friend class to allow it to write LDOS
	 * data. */
	friend class TBTK::DPropertyExtractor;

	/** FileReader is a friend class to allow it to write LDOS data. */
	friend class TBTK::FileReader;
};

inline int Ldos::getDimensions() const{
	return dimensions;
}

inline const int* Ldos::getRanges() const{
	return ranges;
}

inline double Ldos::getLowerBound() const{
	return lowerBound;
}

inline double Ldos::getUpperBound() const{
	return upperBound;
}

inline int Ldos::getResolution() const{
	return resolution;
}

inline int Ldos::getSize() const{
	return size;
}

inline const double* Ldos::getData() const{
	return data;
}

};	//End namespace Property
};	//End namespace TBTK

#endif
