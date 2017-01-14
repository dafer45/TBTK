#ifndef COM_DAFER45_TBTK_LADDER_OPERATOR
#define COM_DAFER45_TBTK_LADDER_OPERATOR

namespace TBTK{

template<typename T>
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
		const FockState<T> &templateState
	);

	/** Destructor. */
	~LadderOperator();

	/** Get type. */
	Type getType() const;

	/** Get associated single-particle state. */
	unsigned int getState() const;

	/** Multiplication operator. */
	FockState<T>& operator*(FockState<T> &rhs) const;
private:
	/** Operator type. */
	Type type;

	/** Single-particle state index. */
	unsigned int state;

	/** State mask. */
	T stateMask;

	/** Least significant bit. */
	T leastSignificantBit;

	/** State corresponding to maximum number of occupied particles. */
	T maxOccupation;
};

template<typename T>
LadderOperator<T>::LadderOperator(
	Type type,
	unsigned int state,
	unsigned int numBitsPerState,
	unsigned int maxOccupation,
	const FockState<T> &templateState
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

	stateMask.print();
	leastSignificantBit.print();
}

template<typename T>
LadderOperator<T>::~LadderOperator(){
}

template<typename T>
typename LadderOperator<T>::Type LadderOperator<T>::getType() const{
	return type;
}

template<typename T>
unsigned int LadderOperator<T>::getState() const{
	return state;
}

template<typename T>
FockState<T>& LadderOperator<T>::operator*(FockState<T> &rhs) const{
	switch(type){
	case Type::Creation:
		rhs.bitRegister += leastSignificantBit;
		break;
	case Type::Annihilation:
		if((rhs.bitRegister & stateMask).toBool()){
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
