#ifndef COM_DAFER45_TBTK_GENERALIZED_GAUGE
#ifndef COM_DAFER45_TBTK_GENERALIZED_GAUGE

namespace TBTK{

class GeneralizedGauge{
public:
	GeneralizedGauge();
	~GeneralizedGauge();
private:
	Index index;
	complex<double> phase
	int sign;
	bool conjugtion;
};

};	//End of namespace TBTK

#endif
