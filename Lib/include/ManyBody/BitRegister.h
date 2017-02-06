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
 *  @file BitRegister.h
 *  @brief Register of bits.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_BIT_REGISTER
#define COM_DAFER45_TBTK_BIT_REGISTER

#include "Streams.h"

namespace TBTK{

class BitRegister{
public:
	/** Constructor. */
	BitRegister(unsigned int numBits = 8*sizeof(unsigned int));

	/** Copy constructor. */
	BitRegister(const BitRegister &bitRegister);

	/** Destructor. */
	~BitRegister();

	/** Bitwise or operator. */
	const BitRegister operator|(const BitRegister &rhs) const;

	/** Bitwise and operator. */
	const BitRegister operator&(const BitRegister &rhs) const;

	/** Bitwise xor operator. */
	const BitRegister operator^(const BitRegister &rhs) const;

	/** Addition operator. */
	const BitRegister operator+(const BitRegister &rhs) const;

	/** Subtraction operator. */
	const BitRegister operator-(const BitRegister &rhs) const;

	/** Less than operator. */
	bool operator<(const BitRegister &rhs) const;

	/** Greater than operator. */
	bool operator>(const BitRegister &rhs) const;

	/** Comparison operator. */
	bool operator==(const BitRegister &rhs) const;

	/** += operator. */
	void operator+=(const BitRegister &rhs);

	/** -= operator. */
	void operator-=(const BitRegister &rhs);

	/** Increment operator. */
	const BitRegister& operator++();

	/** Increment operator. */
	const BitRegister operator++(int);

	/** Decrement operator. */
	const BitRegister& operator--();

	/** Decrement operator. */
	const BitRegister operator--(int);

	/** Assignment operator. */
	void operator=(const BitRegister &rhs);

	/** Assignment operator. */
	void operator=(unsigned int rhs);

	/** Left bitshift operator. */
	BitRegister operator<<(unsigned int rhs) const;

	/** Right bitshift operator. */
	BitRegister operator>>(unsigned int rhs) const;

	/** Set value of bit at a specific position. */
	void setBit(unsigned int position, bool values);

	/** Get bit value. */
	bool getBit(unsigned int position) const;

	/** Set values as unsigned int. */
	void setValues(unsigned int values);

	/** Get values as unsigned int. */
	unsigned int getValues() const;

	/** Returns a bool that is false if all bits are zero, and true
	 *  otherwise. */
	bool toBool() const;

	/** Clear register. */
	void clear();

	/** Print bit register. */
	void print() const;

	/** Returns the number of bits in the register. */
	unsigned int getNumBits() const;

	/** Returns the number of bits that are one. */
	unsigned int getNumOneBits() const;

	/** Returns the most significant bit. */
	bool getMostSignificantBit() const;

	/** Set the most significant bit. */
	void setMostSignificantBit();

	/** Clear the most significant bit. */
	void clearMostSignificantBit();

	/** Create a new BitRegister with the same structure. (Provided to
	 *  ensure the interface is similar with the interface for
	 *  ExtensiveBitRegister.) */
	BitRegister cloneStructure() const;

	/** Return the value as an unsigned int. */
	unsigned int getAsUnsignedInt() const;
private:
	/** Value. */
	unsigned int values;

	/** Mask for the most significant bit. */
	static constexpr unsigned int MOST_SIGNIFICANT_BIT_MASK = 0x1 << (8*sizeof(unsigned int)-1);
};

inline const BitRegister BitRegister::operator|(const BitRegister &rhs) const{
	BitRegister result;
	result.values = values | rhs.values;
	return result;
}

inline const BitRegister BitRegister::operator&(const BitRegister &rhs) const{
	BitRegister result;
	result.values = values & rhs.values;
	return result;
}

inline const BitRegister BitRegister::operator^(const BitRegister &rhs) const{
	BitRegister result;
	result.values = values^rhs.values;
	return result;
}

inline const BitRegister BitRegister::operator+(const BitRegister &rhs) const{
	BitRegister result;
	result.values = values + rhs.values;
	return result;
}

inline const BitRegister BitRegister::operator-(const BitRegister &rhs) const{
	BitRegister result;
	result.values = values - rhs.values;
	return result;
}

inline bool BitRegister::operator<(const BitRegister &rhs) const{
	return values < rhs.values;
}

inline bool BitRegister::operator>(const BitRegister &rhs) const{
	return values > rhs.values;
}

inline bool BitRegister::operator==(const BitRegister &rhs) const{
	return values == rhs.values;
}

inline void BitRegister::operator+=(const BitRegister &rhs){
	values += rhs.values;
}

inline void BitRegister::operator-=(const BitRegister &rhs){
	values -= rhs.values;
}

inline const BitRegister& BitRegister::operator++(){
	values++;
	return *this;
}

inline const BitRegister BitRegister::operator++(int){
	unsigned int returnValue = values;
	values++;
	return returnValue;
}

inline const BitRegister& BitRegister::operator--(){
	values--;
	return *this;
}

inline const BitRegister BitRegister::operator--(int){
	unsigned int returnValue = values;
	values--;
	return returnValue;
}

inline void BitRegister::operator=(const BitRegister &rhs){
	values = rhs.values;
}

inline void BitRegister::operator=(unsigned int rhs){
	values = rhs;
}

inline BitRegister BitRegister::operator<<(unsigned int rhs) const{
	BitRegister result;
	result.values = values << rhs;
	return result;
}

inline BitRegister BitRegister::operator>>(unsigned int rhs) const{
	BitRegister result;
	result.values = values >> rhs;
	return result;
}

inline void BitRegister::print() const{
	for(int n = 8*sizeof(values)-1; n >= 0; n--)
		Streams::out << (0x1 & (values >> n));
	Streams::out << "\n";
}

inline void BitRegister::setBit(unsigned int position, bool value){
	values &= ~(1 << position);
	values ^= (value << position);
}

inline bool BitRegister::getBit(unsigned int position) const{
	return (0x1 & (values >> position));
}

inline void BitRegister::setValues(unsigned int values){
	this->values = values;
}

inline unsigned int BitRegister::getValues() const{
	return values;
}

inline bool BitRegister::toBool() const{
	return values;
}

inline void BitRegister::clear(){
	values = 0;
}

inline unsigned int BitRegister::getNumBits() const{
	return 8*sizeof(values);
}

inline unsigned int BitRegister::getNumOneBits() const{
	unsigned int x = values;
	x = x - ((x >> 1) & 0x55555555);
	x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
	x = (x & 0x0F0F0F0F) + ((x >> 4) & 0x0F0F0F0F);
	x = x + (x >> 8);
	x = x + (x >> 16);
	return (x & 0x0000003F);
}

inline bool BitRegister::getMostSignificantBit() const{
	return values & MOST_SIGNIFICANT_BIT_MASK;
}

inline void BitRegister::setMostSignificantBit(){
	values |= MOST_SIGNIFICANT_BIT_MASK;
}

inline void BitRegister::clearMostSignificantBit(){
	values &= !MOST_SIGNIFICANT_BIT_MASK;
}

inline BitRegister BitRegister::cloneStructure() const{
	return BitRegister(8*sizeof(unsigned int));
}

};	//End of namespace TBTK

#endif
