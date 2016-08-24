/** @package TBTKcalc
 *  @file Magnetization.h
 *  @brief Property container for magnetization
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_MAGNETIZATION
#define COM_DAFER45_TBTK_MAGNETIZATION

#include <complex>

namespace TBTK{
namespace Property{

/** Container for magnetization. */
class Magnetization{
public:
	Magnetization(int rank, const int* dims);
	~Magnetization();

	int getRank();
	const int* getDims();
	int getSize();
	const double* getData();
private:
	int rank;
	int *dims;
	int size;
	std::complex<double> *data;

	friend class CPropertyExtractor;
	friend class DPropertyExtractor;
};

inline int Magnetization::getRank(){
	return rank;
}

inline const int* Magnetization::getDims(){
	return dims;
}

inline int Magnetization::getSize(){
	return size;
}

};	//End namespace Property
};	//End namespace TBTK

#endif
