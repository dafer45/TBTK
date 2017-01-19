#ifndef COM_DAFER45_TBTK_LADDER_OPERATOR
#define COM_DAFER45_TBTK_LADDER_OPERATOR

#include "FockState.h"

namespace TBTK{

template<typename BIT_REGISTER>
class LadderOperator{
public:
	/** Operator type. */
	enum class Type {Creation, Annihilation};

	/** Constructor. */
	LadderOperator(
		Type type,
		unsigned int state,
		unsigned int numBitsPerState,
		unsigned int maxOccupation,
		const FockState<BIT_REGISTER> &templateState
	);

	/** Destructor. */
	~LadderOperator();

	/** Get type. */
	Type getType() const;

	/** Get associated single-particle state. */
	unsigned int getState() const;

	/** Multiplication operator. */
	FockState<BIT_REGISTER>& operator*(FockState<BIT_REGISTER> &rhs) const;
private:
	/** Operator type. */
	Type type;

	/** Single-particle state index. */
	unsigned int state;

	/** State mask. */
	BIT_REGISTER stateMask;

	/** Least significant bit. */
	BIT_REGISTER leastSignificantBit;

	/** State corresponding to maximum number of occupied particles. */
	BIT_REGISTER maxOccupation;
};

template<typename BIT_REGISTER>
LadderOperator<BIT_REGISTER>::LadderOperator(
	Type type,
	unsigned int state,
	unsigned int numBitsPerState,
	unsigned int maxOccupation,
	const FockState<BIT_REGISTER> &templateState
) :
	stateMask(templateState.bitRegister),
	leastSignificantBit(templateState.bitRegister),
	maxOccupation(templateState.bitRegister)
{
	this->type = type;
	this->state = state;

	for(int n = 0; n < stateMask.getNumBits(); n++){
		if(n >= state*numBitsPerState && n < (state+1)*numBitsPerState)
			stateMask.setBit(n, 1);
		else
			stateMask.setBit(n, 0);

		if(n == state*numBitsPerState)
			leastSignificantBit.setBit(n, 1);
		else
			leastSignificantBit.setBit(n, 0);
	}

	this->maxOccupation = maxOccupation;
	this->maxOccupation = (this->maxOccupation << numBitsPerState*state);
}

template<typename BIT_REGISTER>
LadderOperator<BIT_REGISTER>::~LadderOperator(){
}

template<typename BIT_REGISTER>
typename LadderOperator<BIT_REGISTER>::Type LadderOperator<BIT_REGISTER>::getType() const{
	return type;
}

template<typename BIT_REGISTER>
unsigned int LadderOperator<BIT_REGISTER>::getState() const{
	return state;
}

template<typename BIT_REGISTER>
FockState<BIT_REGISTER>& LadderOperator<BIT_REGISTER>::operator*(
	FockState<BIT_REGISTER> &rhs
) const{
	switch(type){
	case Type::Creation:
		if((rhs.bitRegister & stateMask) == maxOccupation){
			rhs.bitRegister.setMostSignificantBit();
			break;
		}
		rhs.bitRegister += leastSignificantBit;
/*		if(!(rhs.bitRegister & stateMask).toBool() || (rhs.bitRegister & stateMask) > maxOccupation)
			rhs.bitRegister.setMostSignificantBit();*/
		break;
	case Type::Annihilation:
		if(!(rhs.bitRegister & stateMask).toBool()){
			rhs.bitRegister.setMostSignificantBit();
			break;
		}
		rhs.bitRegister -= leastSignificantBit;
		break;
	default:
		TBTKExit(
			"LadderOperator<T>::operator*()",
			"This should never happen.",
			"Contact the developer."
		);
	}

	return rhs;
}

};	//End of namespace TBTK

#endif
