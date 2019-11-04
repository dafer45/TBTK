/* Copyright 2016 Kristofer Björnson
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/** @package TBTKcalc
 *  @file ExtensiveBitRegister.h
 *  @brief Register of bits.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_EXTENSIVE_BIT_REGISTER
#define COM_DAFER45_TBTK_EXTENSIVE_BIT_REGISTER

#include "TBTK/Streams.h"
#include "TBTK/TBTKMacros.h"

#include <climits>

namespace TBTK{

/** @brief Register of bits.
 *
 *  The ExtensiveBitRegister is similar to the BitRegister, but allows for
 *  arbitrary number of bits to be stored and operated on.
 *
 *  # Example
 *  \snippet ManyParticle/ExtensiveBitRegister.cpp ExtensiveBitRegister
 *  ## Output
 *  \snippet output/ManyParticle/ExtensiveBitRegister.output ExtensiveBitRegister
 */
class ExtensiveBitRegister{
public:
	/** Constructor. */
	ExtensiveBitRegister();

	/** Constructor. */
	ExtensiveBitRegister(unsigned int numBits);

	/** Copy constructor. */
	ExtensiveBitRegister(const ExtensiveBitRegister &extensiveBitRegister);

	/** Destructor. */
	~ExtensiveBitRegister();

	/** Bitwise or operator. */
	const ExtensiveBitRegister operator|(
		const ExtensiveBitRegister &rhs
	) const;

	/** Bitwise and operator. */
	const ExtensiveBitRegister operator&(
		const ExtensiveBitRegister &rhs
	) const;

	/** Bitwise xor operator. */
	const ExtensiveBitRegister operator^(
		const ExtensiveBitRegister &rhs
	) const;

	/** Addition operator. */
	const ExtensiveBitRegister operator+(
		const ExtensiveBitRegister &rhs
	) const;

	/** Subtraction operator. */
	const ExtensiveBitRegister operator-(
		const ExtensiveBitRegister &rhs
	) const;

	/** Less than operator. */
	bool operator<(const ExtensiveBitRegister &rhs) const;

	/** Greater than operator. */
	bool operator>(const ExtensiveBitRegister &rhs) const;

	/** Comparison operator. */
	bool operator==(const ExtensiveBitRegister &rhs) const;

	/** += operator. */
	void operator+=(const ExtensiveBitRegister &rhs);

	/** -= operator. */
	void operator-=(const ExtensiveBitRegister &rhs);

	/** Increment operator. */
	const ExtensiveBitRegister& operator++();

	/** Increment operator. */
	const ExtensiveBitRegister operator++(int);

	/** Decrement operator. */
	const ExtensiveBitRegister& operator--();

	/** Decrement operator. */
	const ExtensiveBitRegister operator--(int);

	/** Assignment operator. */
	void operator=(const ExtensiveBitRegister &rhs);

	/** Assignment operator. */
	void operator=(const unsigned int rhs);

	/** Left bitshift operator. */
	ExtensiveBitRegister operator<<(unsigned int rhs) const;

	/** Right bitshift operator. */
	ExtensiveBitRegister operator>>(unsigned int rhs) const;

	/** Set value of bit at a specific position. */
	void setBit(unsigned int position, bool values);

	/** Get bit value. */
	bool getBit(unsigned int position) const;

	/** Returns a bool that is false if all bits are zero, and true
	 *  otherwise. */
	bool toBool() const;

	/** Returns an unsigned int containing the least significant bits in
	the register. */
	unsigned int toUnsignedInt() const;

	/** Clear register. */
	void clear();

	/** Print bit register. */
	void print() const;

	/** Return the number of bits in the register. */
	unsigned int getNumBits() const;

	/** Returns the number of bits that are one. */
	unsigned int getNumOneBits() const;

	/** Return the most significant bit. */
	bool getMostSignificantBit() const;

	/** Set the most significant bit. */
	void setMostSignificantBit();

	/** Set the most significant bit. */
	void clearMostSignificantBit();

	/** Create a new ExtensiveBitRegister with the same strucutre. */
	ExtensiveBitRegister cloneStructure() const;

	/** Get string representation of the ExtensiveBitRegister.
	 *
	 *  @return A string representation of the ExtensiveBitRegister. */
	std::string toString() const;

	/** Writes the ExtensiveBitRegister toString-representation to a
	 *  stream.
	 *
	 *  @param stream The stream to srite to.
	 *  @param extensiveBitRegister The ExtensiveBitRegister to write.
	 *
	 *  @return Reference to the output stream just written to. */
	friend std::ostream& operator<<(
		std::ostream &stream,
		const ExtensiveBitRegister &extensiveBitRegister
	);
private:
	/** Size of values array. */
	unsigned int size;

	/** Value. */
	unsigned int *values;

	/** Mask for the most significant bit. */
	static constexpr unsigned int MOST_SIGNIFICANT_BIT_MASK
		= (unsigned int)0x1 << (8*sizeof(unsigned int) - 1);
};

inline const ExtensiveBitRegister ExtensiveBitRegister::operator|(
	const ExtensiveBitRegister &rhs
) const{
	TBTKAssert(
		size == rhs.size,
		"ExtensiveBitRegister::operator|()",
		"Incompatible register sizes.",
		""
	);

	ExtensiveBitRegister result = cloneStructure();
	for(unsigned int n = 0; n < size; n++)
		result.values[n] = values[n] | rhs.values[n];

	return result;
}

inline const ExtensiveBitRegister ExtensiveBitRegister::operator&(
	const ExtensiveBitRegister &rhs
) const{
	TBTKAssert(
		size == rhs.size,
		"ExtensiveBitRegister::operator|()",
		"Incompatible register sizes.",
		""
	);

	ExtensiveBitRegister result = cloneStructure();
	for(unsigned int n = 0; n < size; n++)
		result.values[n] = values[n] & rhs.values[n];

	return result;
}

inline const ExtensiveBitRegister ExtensiveBitRegister::operator^(
	const ExtensiveBitRegister &rhs
) const{
	TBTKAssert(
		size == rhs.size,
		"ExtensiveBitRegister::operator|()",
		"Incompatible register sizes.",
		""
	);

	ExtensiveBitRegister result = cloneStructure();
	for(unsigned int n = 0; n < size; n++)
		result.values[n] = values[n] ^ rhs.values[n];

	return result;
}

inline const ExtensiveBitRegister ExtensiveBitRegister::operator+(
	const ExtensiveBitRegister &rhs
) const{
	TBTKAssert(
		size == rhs.size,
		"ExtensiveBitRegister::operator+()",
		"Incompatible register sizes.",
		""
	);

	ExtensiveBitRegister result = cloneStructure();
	unsigned int carry = 0;
	for(unsigned int n = 0; n < size; n++){
		result.values[n] = values[n] + rhs.values[n];
		unsigned int newCarry = (result.values[n] < values[n]);
		result.values[n] += carry;
		carry = newCarry | (result.values[n] < values[n]);
	}

	return result;
}

inline const ExtensiveBitRegister ExtensiveBitRegister::operator-(
	const ExtensiveBitRegister &rhs
) const{
	TBTKAssert(
		size == rhs.size,
		"ExtensiveBitRegister::operator+()",
		"Incompatible register sizes.",
		""
	);

	ExtensiveBitRegister result = cloneStructure();
	unsigned int carry = 0;
	for(unsigned int n = 0; n < size; n++){
		result.values[n] = values[n] - rhs.values[n];
		unsigned int newCarry = (result.values[n] > values[n]);
		result.values[n] -= carry;
		carry = newCarry | (result.values[n] > values[n]);
	}

	return result;
}

inline bool ExtensiveBitRegister::operator<(
	const ExtensiveBitRegister &rhs
) const{
	for(int n = size-1; n >= 0; n--){
		if(values[n] < rhs.values[n])
			return true;
		if(values[n] > rhs.values[n])
			return false;
	}

	return false;
}

inline bool ExtensiveBitRegister::operator>(
	const ExtensiveBitRegister &rhs
) const{
	for(int n = size-1; n >= 0; n--){
		if(values[n] > rhs.values[n])
			return true;
		if(values[n] < rhs.values[n])
			return false;
	}

	return false;
}

inline bool ExtensiveBitRegister::operator==(
	const ExtensiveBitRegister &rhs
) const{
	for(unsigned int n = 0; n < size; n++){
		if(values[n] != rhs.values[n])
			return false;
	}

	return true;
}

inline void ExtensiveBitRegister::operator+=(const ExtensiveBitRegister &rhs){
	TBTKAssert(
		size == rhs.size,
		"ExtensiveBitRegister::operator+=()",
		"Incompatible register sizes.",
		""
	);

	unsigned int carry = 0;
	for(unsigned int n = 0; n < size; n++){
		unsigned int temp = values[n];
		values[n] = temp + rhs.values[n];
		unsigned int newCarry = (values[n] < temp);
		values[n] += carry;
		carry = newCarry | (values[n] < temp);
	}
}

inline void ExtensiveBitRegister::operator-=(const ExtensiveBitRegister &rhs){
	TBTKAssert(
		size == rhs.size,
		"ExtensiveBitRegister::operator-=()",
		"Incompatible register sizes.",
		""
	);

	unsigned int carry = 0;
	for(unsigned int n = 0; n < size; n++){
		unsigned int temp = values[n];
		values[n] = temp - rhs.values[n];
		unsigned int newCarry = (values[n] > temp);
		values[n] -= carry;
		carry = newCarry | (values[n] > temp);
	}
}

inline const ExtensiveBitRegister& ExtensiveBitRegister::operator++(){
	for(unsigned int n = 0; n < size; n++){
		if(values[n] == UINT_MAX){
			values[n]++;
		}
		else{
			values[n]++;
			break;
		}
	}

	return *this;
}

inline const ExtensiveBitRegister ExtensiveBitRegister::operator++(int){
	ExtensiveBitRegister returnValue(*this);
	++(*this);

	return returnValue;
}

inline const ExtensiveBitRegister& ExtensiveBitRegister::operator--(){
	for(unsigned int n = 0; n < size; n++){
		if(values[n] == 0){
			values[n]--;
		}
		else{
			values[n]--;
			break;
		}
	}

	return *this;
}

inline const ExtensiveBitRegister ExtensiveBitRegister::operator--(int){
	ExtensiveBitRegister returnValue(*this);
	--(*this);

	return returnValue;
}

inline void ExtensiveBitRegister::operator=(const ExtensiveBitRegister &rhs){
	if(this != &rhs){
		TBTKAssert(
			size == rhs.size,
			"ExtensiveBitRegister::operator=()",
			"Incompatible register sizes.",
			""
		);

		for(unsigned int n = 0; n < size; n++)
			values[n] = rhs.values[n];
	}
}

inline void ExtensiveBitRegister::operator=(const unsigned int rhs){
	values[0] = rhs;
	for(unsigned int n = 1; n < size; n++)
		values[n] = 0;
}

inline ExtensiveBitRegister ExtensiveBitRegister::operator<<(unsigned int rhs) const{
	ExtensiveBitRegister result(*this);
	if(rhs > 8*sizeof(unsigned int)){
		result = result << (rhs - 8*sizeof(unsigned int));

		for(int n = size-1; n > 0; n--)
			result.values[n] = result.values[n-1];
		result.values[0] = 0;
	}
	else if(rhs == 8*sizeof(unsigned int)){
		for(int n = size-1; n > 0; n--)
			result.values[n] = result.values[n-1];
		result.values[0] = 0;
	}
	else if(rhs != 0){
		for(int n = size-1; n >= 0; n--){
			result.values[n] = result.values[n] << rhs;
			if(n > 0)
				result.values[n] |= result.values[n-1] >> (8*sizeof(unsigned int) - rhs);
		}
	}

	return result;
}

inline ExtensiveBitRegister ExtensiveBitRegister::operator>>(unsigned int rhs) const{
	ExtensiveBitRegister result(*this);
	if(rhs > 8*sizeof(unsigned int)){
		result = result >> (rhs - 8*sizeof(unsigned int));

		for(unsigned int n = 0; n < size-1; n++)
			result.values[n] = result.values[n+1];
		result.values[size-1] = 0;
	}
	else if(rhs != 0){
		for(unsigned int n = 0; n < size; n++){
			result.values[n] = result.values[n] >> rhs;
			if(n < size-1)
				result.values[n] |= result.values[n+1] << (8*sizeof(unsigned int) - rhs);
		}
	}

	return result;
}

inline void ExtensiveBitRegister::print() const{
	for(unsigned int n = size; n > 0; n--){
		for(int c = 8*sizeof(unsigned int)-1; c >= 0; c--)
			Streams::out << (0x1 & (values[n-1] >> c));
		Streams::out << " ";
	}
	Streams::out << "\n";
}

inline void ExtensiveBitRegister::setBit(unsigned int position, bool value){
	values[position/(8*sizeof(unsigned int))] &= ~(1 << (position%(8*sizeof(unsigned int))));
	values[position/(8*sizeof(unsigned int))] ^= (value << (position%(8*sizeof(unsigned int))));
}

inline bool ExtensiveBitRegister::getBit(unsigned int position) const{
	return (0x1 & (values[position/(8*sizeof(unsigned int))] >> (position%(8*sizeof(unsigned int)))));
}

inline bool ExtensiveBitRegister::toBool() const{
	for(unsigned int n = 0; n < size; n++)
		if(values[n])
			return true;

	return false;
}

inline unsigned int ExtensiveBitRegister::toUnsignedInt() const{
	return values[0];
}

inline void ExtensiveBitRegister::clear(){
	for(unsigned int n = 0; n < size; n++)
		values[n] = 0;
}

inline unsigned int ExtensiveBitRegister::getNumBits() const{
	return size*8*sizeof(unsigned int);
}

inline unsigned int ExtensiveBitRegister::getNumOneBits() const{
	unsigned int numOnes = 0;
	for(unsigned int n = 0; n < size; n++){
		unsigned int x = values[n];
		x = x - ((x >> 1) & 0x55555555);
		x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
		x = (x & 0x0F0F0F0F) + ((x >> 4) & 0x0F0F0F0F);
		x = x + (x >> 8);
		x = x + (x >> 16);
		numOnes += (x & 0x0000003F);
	}

	return numOnes;
}

inline bool ExtensiveBitRegister::getMostSignificantBit() const{
	return values[size-1] & MOST_SIGNIFICANT_BIT_MASK;
}

inline void ExtensiveBitRegister::setMostSignificantBit(){
	values[size-1] |= MOST_SIGNIFICANT_BIT_MASK;
}

inline void ExtensiveBitRegister::clearMostSignificantBit(){
	values[size-1] &= !MOST_SIGNIFICANT_BIT_MASK;
}

inline ExtensiveBitRegister ExtensiveBitRegister::cloneStructure() const{
	return ExtensiveBitRegister(size*8*sizeof(unsigned int));
}

inline std::string ExtensiveBitRegister::toString() const{
	std::stringstream stream;
	for(unsigned int n = getNumBits(); n > 0; n--)
		stream << getBit(n - 1);

	return stream.str();
}

inline std::ostream& operator<<(
	std::ostream &stream,
	const ExtensiveBitRegister &extensiveBitRegister
){
	stream << extensiveBitRegister.toString();

	return stream;
}

};	//End of namespace TBTK

#endif
