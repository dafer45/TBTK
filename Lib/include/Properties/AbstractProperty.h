/* Copyright 2017 Kristofer Björnson
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
 *  @file AbstractPoperty.h
 *  @brief Abstract Property class
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_ABSTRACT_PROPERTY
#define COM_DAFER45_TBTK_ABSTRACT_PROPERTY

#include "IndexDescriptor.h"

namespace TBTK{
namespace Property{

template<typename DataType>
class AbstractProperty{
public:
	/** Get block size. */
	unsigned int getBlockSize() const;

	/** Set size. */
	void setSize(unsigned int size);

	/** Get size. */
	unsigned int getSize() const;

	/** Get data. */
	const DataType* getData() const;

	/** Same as getData, but with write access. */
	DataType* getDataRW();

	/** Get size in bytes. */
	unsigned int getSizeInBytes() const;

	/** Returns data on raw format. Intended for use in serialization. */
	char* getRawData();

	/** Get the dimension of the data. */
	unsigned int getDimensions() const;

	/** Get the ranges for the dimensions of the density. */
	const int* getRanges() const;

	/** Get the offset in memory for given Index. */
	unsigned int getOffset(const Index &index) const;

	/** Get IndexDescriptor. */
	const IndexDescriptor& getIndexDescriptor() const;

	/** Returns the storage format. */
	IndexDescriptor::Format getFormat() const;

	/** Returns true if the property contains data for the given index. */
	bool contains(const Index &index) const;

	/** Function call operator. */
	virtual DataType operator()(const Index &index, unsigned int offset = 0) const;

	/** Function call operator. */
	virtual DataType operator()(unsigned int offset) const;
protected:
	/** Constructor. */
	AbstractProperty(
		unsigned int blockSize
	);

	/** Constructor. */
	AbstractProperty(
		unsigned int blockSize,
		const DataType *data
	);

	/** Constructor. */
	AbstractProperty(
		unsigned int dimensions,
		const int *ranges,
		unsigned int blockSize
	);

	/** Constructor. */
	AbstractProperty(
		unsigned int dimensions,
		const int *ranges,
		unsigned int blockSize,
		const DataType *data
	);

	/** Constructor. */
	AbstractProperty(
		const IndexTree &indexTree,
		unsigned int blockSize
	);

	/** Constructor. */
	AbstractProperty(
		const IndexTree &indexTree,
		unsigned int blockSize,
		const DataType *data
	);

	/** Copy constructor. */
	AbstractProperty(const AbstractProperty &abstractProperty);

	/** Move constructor. */
	AbstractProperty(AbstractProperty &&abstractProperty);

	/** Destructor. */
	virtual ~AbstractProperty();

	/** Assignment operator. */
	AbstractProperty& operator=(const AbstractProperty &abstractProperty);

	/** Move assignment operator. */
	AbstractProperty& operator=(AbstractProperty &&abstractProperty);
private:
	/** IndexDescriptor describing the memory layout of the data. */
	IndexDescriptor indexDescriptor;

	/** Size of a single block of data needed to describe a single point
	 *  refered to by an index. In particular:
	 *  indexDescriptor.size()*blockSize = size. */
	unsigned int blockSize;

	/** Number of data elements. */
	unsigned int size;

	/** Data. */
	DataType *data;
};

template<typename DataType>
inline unsigned int AbstractProperty<DataType>::getBlockSize() const{
	return blockSize;
}

template<typename DataType>
inline void AbstractProperty<DataType>::setSize(unsigned int size){
	this->size = size;
	if(data != nullptr)
		delete [] data;
	data = new DataType[size];
}

template<typename DataType>
inline unsigned int AbstractProperty<DataType>::getSize() const{
	return size;
}

template<typename DataType>
inline const DataType* AbstractProperty<DataType>::getData() const{
	return data;
}

template<typename DataType>
inline DataType* AbstractProperty<DataType>::getDataRW(){
	return data;
}

template<typename DataType>
inline unsigned int AbstractProperty<DataType>::getSizeInBytes() const{
	return size*sizeof(DataType);
}

template<typename DataType>
inline char* AbstractProperty<DataType>::getRawData(){
	return (char*)data;
}

template<typename DataType>
inline unsigned int AbstractProperty<DataType>::getDimensions() const{
	return indexDescriptor.getDimensions();
}

template<typename DataType>
inline const int* AbstractProperty<DataType>::getRanges() const{
	return indexDescriptor.getRanges();
}

template<typename DataType>
inline unsigned int AbstractProperty<DataType>::getOffset(
	const Index &index
) const{
	return blockSize*indexDescriptor.getLinearIndex(index);
}

template<typename DataType>
inline const IndexDescriptor& AbstractProperty<DataType>::getIndexDescriptor() const{
	return indexDescriptor;
}

template<typename DataType>
inline IndexDescriptor::Format AbstractProperty<DataType>::getFormat() const{
	return indexDescriptor.getFormat();
}

template<typename DataType>
inline bool AbstractProperty<DataType>::contains(const Index &index) const{
	return indexDescriptor.contains(index);
}

template<typename DataType>
inline DataType AbstractProperty<DataType>::operator()(const Index &index, unsigned int offset) const{
	return data[getOffset(index) + offset];
}

template<typename DataType>
inline DataType AbstractProperty<DataType>::operator()(unsigned int offset) const{
	return data[offset];
}

template<typename DataType>
AbstractProperty<DataType>::AbstractProperty(
	unsigned int blockSize
) :
	indexDescriptor(IndexDescriptor::Format::None)
{
	this->blockSize = blockSize;

	size = blockSize;
	data = new DataType[size];
	for(unsigned int n = 0; n < size; n++)
		data[n] = 0.;
}

template<typename DataType>
AbstractProperty<DataType>::AbstractProperty(
	unsigned int blockSize,
	const DataType *data
) :
	indexDescriptor(IndexDescriptor::Format::None)
{
	this->blockSize = blockSize;

	size = blockSize;
	this->data = new DataType[size];
	for(unsigned int n = 0; n < size; n++)
		this->data[n] = data[n];
}

template<typename DataType>
AbstractProperty<DataType>::AbstractProperty(
	unsigned int dimensions,
	const int *ranges,
	unsigned int blockSize
) :
	indexDescriptor(IndexDescriptor::Format::Ranges)
{
	this->blockSize = blockSize;

	indexDescriptor.setDimensions(dimensions);
	int *thisRanges = indexDescriptor.getRanges();
	for(unsigned int n = 0; n < dimensions; n++)
		thisRanges[n] = ranges[n];

	size = blockSize*indexDescriptor.getSize();
	data = new DataType[size];
	for(unsigned int n = 0; n < size; n++)
		data[n] = 0.;
}

template<typename DataType>
AbstractProperty<DataType>::AbstractProperty(
	unsigned int dimensions,
	const int *ranges,
	unsigned int blockSize,
	const DataType *data
) :
	indexDescriptor(IndexDescriptor::Format::Ranges)
{
	this->blockSize = blockSize;

	indexDescriptor.setDimensions(dimensions);
	int *thisRanges = indexDescriptor.getRanges();
	for(unsigned int n = 0; n < dimensions; n++)
		thisRanges[n] = ranges[n];

	size = blockSize*indexDescriptor.getSize();
	this->data = new DataType[size];
	for(unsigned int n = 0; n < size; n++)
		this->data[n] = data[n];
}

template<typename DataType>
AbstractProperty<DataType>::AbstractProperty(
	const IndexTree &indexTree,
	unsigned int blockSize
) :
	indexDescriptor(IndexDescriptor::Format::Custom)
{
	this->blockSize = blockSize;

	indexDescriptor.setIndexTree(indexTree);

	size = blockSize*indexDescriptor.getSize();
	data = new DataType[size];
	for(unsigned int n = 0; n < size; n++)
		data[n] = 0.;
}

template<typename DataType>
AbstractProperty<DataType>::AbstractProperty(
	const IndexTree &indexTree,
	unsigned int blockSize,
	const DataType *data
) :
	indexDescriptor(IndexDescriptor::Format::Custom)
{
	this->blockSize = blockSize;

	indexDescriptor.setIndexTree(indexTree);

	size = blockSize*indexDescriptor.getSize();
	this->data = new DataType[size];
	for(unsigned int n = 0; n < size; n++)
		this->data[n] = data[n];
}

template<typename DataType>
AbstractProperty<DataType>::AbstractProperty(
	const AbstractProperty &abstractProperty
) :
	indexDescriptor(abstractProperty.indexDescriptor)
{
	blockSize = abstractProperty.blockSize;

	size = abstractProperty.size;
	if(abstractProperty.data == nullptr){
		data = nullptr;
	}
	else{
		data = new DataType[size];
		for(unsigned int n = 0; n < size; n++)
			data[n] = abstractProperty.data[n];
	}
}

template<typename DataType>
AbstractProperty<DataType>::AbstractProperty(
	AbstractProperty &&abstractProperty
) :
	indexDescriptor(std::move(abstractProperty.indexDescriptor))
{
	blockSize = abstractProperty.blockSize;

	size = abstractProperty.size;
	if(abstractProperty.data == nullptr){
		data = nullptr;
	}
	else{
		data = abstractProperty.data;
		abstractProperty.data = nullptr;
	}
}

template<typename DataType>
AbstractProperty<DataType>::~AbstractProperty(){
	if(data != nullptr)
		delete [] data;
}

template<typename DataType>
AbstractProperty<DataType>& AbstractProperty<DataType>::operator=(
	const AbstractProperty &rhs
){
	if(this != &rhs){
		indexDescriptor = rhs.indexDescriptor;

		blockSize = rhs.blockSize;

		size = rhs.size;
		if(rhs.data == nullptr){
			data = nullptr;
		}
		else{
			data = new DataType[size];
			for(unsigned int n = 0; n < size; n++)
				data[n] = rhs.data[n];
		}
	}

	return *this;
}

template<typename DataType>
AbstractProperty<DataType>& AbstractProperty<DataType>::operator=(
	AbstractProperty &&rhs
){
	if(this != &rhs){
		indexDescriptor = std::move(rhs.indexDescriptor);

		blockSize = rhs.blockSize;

		size = rhs.size;
		if(rhs.data == nullptr){
			data = nullptr;
		}
		else{
			data = rhs.data;
			rhs.data = nullptr;
		}
	}

	return *this;
}

};	//End namespace Property
};	//End namespace TBTK

#endif
