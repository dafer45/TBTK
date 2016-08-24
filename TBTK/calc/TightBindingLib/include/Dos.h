/** @package TBTKcalc
 *  @file Dos.h
 *  @brief Property container for density of states (DOS)
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_DOS
#define COM_DAFER45_TBTK_DOS

namespace TBTK{
namespace Property{

/** Container for density of states (DOS). */
class Dos{
public:
	Dos(double lowerLimit, double upperLimit, int resolution);
	~Dos();

	double getLowerLimit();
	double getUpperLimit();
	int getResolution();
	const double* getData();
private:
	double lowerLimit;
	double upperLimit;
	int resolution;
	double *data;

	friend class CPropertyExtractor;
	friend class DPropertyExtractor;
};

inline double Dos::getLowerLimit(){
	return lowerLimit;
}

inline double Dos::getUpperLimit(){
	return upperLimit;
}

inline int Dos::getResolution(){
	return resolution;
}

inline const double* Dos::getData(){
	return data;
}

};	//End namespace Property
};	//End namespace TBTK

#endif
