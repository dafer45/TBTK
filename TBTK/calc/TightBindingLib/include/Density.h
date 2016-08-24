/** @package TBTKcalc
 *  @file Density.h
 *  @brief Property container for density
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_DENSITY
#define COM_DAFER45_TBTK_DENSITY

namespace TBTK{
namespace Property{

/** Container for density. */
class Density{
public:
	Density(int rank, const int *dims);
	~Density();

	int getRank();
	const int* getDims();
	int getSize();
	const double* getData();
private:
	int rank;
	int *dims;
	int size;
	double *data;

	friend class CPropertyExtractor;
	friend class DPropertyExtractor;
};

inline int Density::getRank(){
	return rank;
}

inline const int* Density::getDims(){
	return dims;
}

inline int Density::getSize(){
	return size;
}

inline const double* Density::getData(){
	return data;
}

};	//End namespace Property
};	//End namespace TBTK

#endif
