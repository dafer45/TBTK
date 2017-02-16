#ifndef COM_DAFER45_TBTK_INTERACTION_AMPLITUDE_SET
#define COM_DAFER45_TBTK_INTERACTION_AMPLITUDE_SET

#include "HoppingAmplitudeSet.h"
#include "InteractionAmplitude.h"

#include <complex>

namespace TBTK{

class InteractionAmplitudeSet{
public:
	/** Constructor. */
	InteractionAmplitudeSet(const HoppingAmplitudeSet *hoppingAmplitudeSet);

	/** Destructor. */
	~InteractionAmplitudeSet();

	/** Add interaction. */
	void addIA(InteractionAmplitude ia);

	/** Returns the number of interaction amplitudes. */
	unsigned int getNumInteractionAmplitudes() const;

	/** Returns an interaction amplitude. */
	const InteractionAmplitude& getInteractionAmplitude(
		unsigned int n
	) const;
private:
	/** Single-particle HoppingAmplitudeSet. */
	const HoppingAmplitudeSet *hoppingAmplitudeSet;

	/** Interaction amplitudes. */
	std::vector<InteractionAmplitude> interactionAmplitudes;
};

inline void InteractionAmplitudeSet::addIA(InteractionAmplitude ia){
	interactionAmplitudes.push_back(ia);
}

inline unsigned int InteractionAmplitudeSet::getNumInteractionAmplitudes() const{
	return interactionAmplitudes.size();
}

inline const InteractionAmplitude& InteractionAmplitudeSet::getInteractionAmplitude(
	unsigned int n
) const{
	return interactionAmplitudes.at(n);
}

};	//End of namespace TBTK

#endif
