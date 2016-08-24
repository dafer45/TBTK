/** @package TBTKcalc
 *  @file Ldos.h
 *  @brief Property container for local density of states (LDOS)
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_LDOS
#define COM_DAFER45_TBTK_LDOS

namespace TBTK{
namespace Property{

/** Container for local density of states (LDOS). */
class Ldos{
public:
	Ldos(int rank, const int *dims, double lowerLimit, double upperLimit, int resolution);
	~Ldos();

	int getRank();
	const int* getDims();
	double getLowerLimit();
	double getUpperLimit();
	int getResolution();
	int getSize();
	const double* getData();
private:
	int rank;
	int *dims;
	double lowerLimit;
	double upperLimit;
	int resolution;
	int size;
	double *data;

	friend class CPropertyExtractor;
	friend class DPropertyExtractor;
};

inline int Ldos::getRank(){
	return rank;
}

inline const int* Ldos::getDims(){
	return dims;
}

inline double Ldos::getLowerLimit(){
	return lowerLimit;
}

inline double Ldos::getUpperLimit(){
	return upperLimit;
}

inline int Ldos::getResolution(){
	return resolution;
}

inline int Ldos::getSize(){
	return size;
}

inline const double* Ldos::getData(){
	return data;
}

};	//End namespace Property
};	//End namespace TBTK

#endif
