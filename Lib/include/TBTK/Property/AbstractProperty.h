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
 *  @brief Abstract Property class.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_ABSTRACT_PROPERTY
#define COM_DAFER45_TBTK_ABSTRACT_PROPERTY

#include "TBTK/Property/IndexDescriptor.h"
#include "TBTK/SpinMatrix.h"
#include "TBTK/TBTKMacros.h"

#include "TBTK/json.hpp"

namespace TBTK{
namespace Property{

/** @brief Abstract Property class.
 *
 *  The AbstractProperty provides a generic storage for data of different type
 *  and storage structure and enables the implementation of specific
 *  Properties. To understand the storage structure, it is important to know
 *  that the Property structures data in several layers and that each layer is
 *  customizeable to allow for Properties with relatively different structure
 *  to be stored.
 *
 *  <b>DataType:</b><br/>
 *  The first layer allows for the data type of the individual data elements to
 *  be customized through the template argument DataType.
 *
 *  <b>Block:</b><br/>
 *  In the next layer data is grouped into blocks of N elements. This allows
 *  for Properties to be grouped without specific knowledge about what the
 *  group structure originates from. It is up to the individual Properties to
 *  give meaning to the internal structure of the block, but a common usage
 *  case is to store an energy resolved property for a number of energies.
 *
 *  <b>Index structure:</b><br/>
 *  In the third layer data blocks are given @link Index Indices \endlink. For
 *  flexibility and the ability to optimize for different use cases, several
 *  different storage formats are available for the Index structure and
 *  internally an IndexDescriptor is used to handle the differnt formats. These
 *  formats are:
 *  <br/><br/>
 *  <i>IndexDescriptor::Format::None</i>:<br/>
 *  Used when the Property has no Index structure.
 *  <br/><br/>
 *  <i>IndexDescriptor::Format::Ranges</i>:<br/>
 *  Used to store a Property for a (possibly multi-dimensional) range of @link
 *  Index Indices @endlink. The Ranges format is particularly efficient and
 *  imposes a regular grid structure on the data that sometimes can be required
 *  to for example plotting the Property. However, it has the limitation that
 *  it cannot be used to store Properties for a few select points on a grid, or
 *  for an Index strcucture that does not have a particular grid structure.
 *  <br/><br/>
 *  IndexDescriptor::Format::Custom:<br/>
 *  The Custom format has the benefit of being able to store the Property for a
 *  custom selected set of @link Index Indices @endlink. It also allows the
 *  value of the Property to be accessed using the function notation
 *  property(index) or property(index, n). Where index is and Index and n is a
 *  number indexing into the data block for the given Index. To achieve this
 *  flexibility the Custom format comes with a slightly larger overhead than
 *  the Ranges format.
 *  <br/><br/>
 *  Sometimes it is usefull to input or access data in raw format, especially
 *  when writing custom Property classes or PropertyExtractors. In this case it
 *  is important to know that internally the data is stored in a single
 *  continous DataType array with blocks stored sequentially. The order the
 *  blocks are stored in when containing data for multiple @link Index Indices
 *  @endlink is such that if the @link Index Indices@endlink are added to an
 *  IndexTree, they appear in the order of the corresponding linear indices
 *  obtained from the IndexTree. */
template<
	typename DataType,
	bool isFundamental = std::is_fundamental<DataType>::value,
	bool isSerializable = std::is_base_of<Serializable, DataType>::value
>
class AbstractProperty : public Serializable{
public:
	/** Get block size.
	 *
	 *  @return The size per block. */
	unsigned int getBlockSize() const;

	/** Get size.
	 *
	 *  @return The total number of data elements in the Property. I.e. the
	 *  number of blocks times the block size. */
	unsigned int getSize() const;

	/** Get data.
	 *
	 *  @return The data on the raw format described in the detailed
	 *  description. */
	const DataType* getData() const;

	/** Get data. Same as AbstractProperty::getData(), but with write
	 *  access.
	 *
	 *  @return The data on the raw format described in the detailed
	 *  description. */
	DataType* getDataRW();

	/** Get size in bytes. */
//	unsigned int getDataSizeInBytes() const;

	/** Returns data on raw format. Intended for use in serialization. */
//	char* getRawData();

	/** Get the dimension of the data. */
	unsigned int getDimensions() const;

	/** Get the ranges for the dimensions of the density. */
//	const int* getRanges() const;
	std::vector<int> getRanges() const;

	/** Get the offset in memory for given Index. */
	int getOffset(const Index &index) const;

	/** Get IndexDescriptor. */
	const IndexDescriptor& getIndexDescriptor() const;

	/** Returns true if the property contains data for the given index. */
	bool contains(const Index &index) const;

	/** Function call operator. */
	virtual const DataType& operator()(const Index &index, unsigned int offset = 0) const;

	/** Function call operator. */
	virtual DataType& operator()(const Index &index, unsigned int offset = 0);

	/** Function call operator. */
	virtual const DataType& operator()(unsigned int offset) const;

	/** Function call operator. */
	virtual DataType& operator()(unsigned int offset);

	/** Set whether access of index not contained in the Property is
	 *  allowed or not. If eneabled, remember to also initialize the value
	 *  used for out of bounds access using
	 *  AbstractProperty::setDefaultValue(). */
	void setAllowIndexOutOfBoundsAccess(bool allowIndexOutOfBoundsAccess);

	/** Set the value that is returned when accessing indices not contained
	 *  in the Property. Only use if
	 *  AbstractProperty::setAllowIndexOutOfBoundsAccess(true) is called. */
	void setDefaultValue(const DataType &defaultValue);

	/** Implements Serializable::serialize(). */
	virtual std::string serialize(Mode mode) const;
protected:
	/** Constructs an uninitialized AbstractProperty. */
	AbstractProperty();

	/** Constructs an AbstractProperty with a single data block (i.e. no
	 *  Index structure).
	 *
	 *  @param blockSize The number of data elements in the block. */
	AbstractProperty(unsigned int blockSize);

	/** Constructs an AbstractProperty with a single data block (i.e. no
	 *  Index structure) and fills the internal memory with the data
	 *  provided as input.
	 *
	 *  @param blockSize The number of data elements in the block.
	 *  @param data The data stored on the raw format described in the
	 *  detailed description of the class. */
	AbstractProperty(
		unsigned int blockSize,
		const DataType *data
	);

	/** Constructs an AbstractProperty with the Ranges format.
	 *
	 *  @param dimensions Number of dimensons for the grid.
	 *  @param ranges A list of upper limits for the ranges of the
	 *  different dimensions. The nth dimension will have the range [0,
	 *  ranges[n]).
	 *
	 *  @param blockSize The number of data elements per block. */
	AbstractProperty(
		unsigned int dimensions,
		const int *ranges,
		unsigned int blockSize
	);

	/** Constructs an AbstractProperty with the Ranges format and fills the
	 *  internal memory with the data provided as input.
	 *
	 *  @param dimensions Number of dimensons for the grid.
	 *  @param ranges A list of upper limits for the ranges of the
	 *  different dimensions. The nth dimension will have the range [0,
	 *  ranges[n]).
	 *
	 *  @param blockSize The number of data elements per block.
	 *  @param data The data stored on the raw format described in the
	 *  detailed description of the class. */
	AbstractProperty(
		unsigned int dimensions,
		const int *ranges,
		unsigned int blockSize,
		const DataType *data
	);

	/** Constructs and AbstractProperty with the Custom format.
	 *
	 *  @param indexTree IndexTree containing the @link Index Indices
	 *  @endlink that the AbstractProperty should contain data for.
	 *
	 *  @param blockSize The number of data elements per block. */
	AbstractProperty(
		const IndexTree &indexTree,
		unsigned int blockSize
	);

	/** Constructs and AbstractProperty with the Custom format.
	 *
	 *  @param indexTree IndexTree containing the @link Index Indices
	 *  @endlink that the AbstractProperty should contain data for.
	 *
	 *  @param blockSize The number of data elements per block.
	 *  @param data The data stored on the raw format described in the
	 *  detailed description of the class. */
	AbstractProperty(
		const IndexTree &indexTree,
		unsigned int blockSize,
		const DataType *data
	);

	/** Copy constructor.
	 *
	 *  @param abstractProperty AbstractProperty to copy. */
	AbstractProperty(const AbstractProperty &abstractProperty);

	/** Move constructor.
	 *
	 *  @param abstractProperty AbstractProperty to move. */
	AbstractProperty(AbstractProperty &&abstractProperty);

	/** Constructor. Constructs the AbstractProperty from a serialization
	 *  string.
	 *
	 *  @param serialization Serialization string from which to construct
	 *  the AbstractProperty.
	 *
	 *  @param mode Mode with which the string has been serialized. */
	AbstractProperty(
		const std::string &serialization,
		Mode mode
	);

	/** Destructor. */
	virtual ~AbstractProperty();

	/** Assignment operator.
	 *
	 *  @param rhs AbstractProperty to assign to the left hand side.
	 *
	 *  @return Reference to the assigned AbstractProperty. */
	AbstractProperty& operator=(const AbstractProperty &abstractProperty);

	/** Move assignment operator.
	 *
	 *  @param rhs AbstractProperty to assign to the left hand side.
	 *
	 *  @return Reference to the assigned AbstractProperty. */
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

	/** Flag indicating whether access of */
	bool allowIndexOutOfBoundsAccess;

	/** Default value used for out of bounds access. */
	DataType defaultValue;
};

template<typename DataType, bool isFundamental, bool isSerializable>
inline unsigned int AbstractProperty<
	DataType,
	isFundamental,
	isSerializable
>::getBlockSize() const{
	return blockSize;
}

template<typename DataType, bool isFundamental, bool isSerializable>
inline unsigned int AbstractProperty<
	DataType,
	isFundamental,
	isSerializable
>::getSize() const{
	return size;
}

template<typename DataType, bool isFundamental, bool isSerializable>
inline const DataType* AbstractProperty<
	DataType,
	isFundamental,
	isSerializable
>::getData() const{
	return data;
}

template<typename DataType, bool isFundamental, bool isSerializable>
inline DataType* AbstractProperty<
	DataType,
	isFundamental,
	isSerializable
>::getDataRW(){
	return data;
}

/*template<typename DataType, bool isFundamental, bool isSerializable>
inline unsigned int AbstractProperty<
	DataType,
	isFundamental,
	isSerializable
>::getDataSizeInBytes() const{
	return size*sizeof(DataType);
}

template<typename DataType, bool isFundamental, bool isSerializable>
inline char* AbstractProperty<
	DataType,
	isFundamental,
	isSerializable
>::getRawData(){
	return (char*)data;
}*/

template<typename DataType, bool isFundamental, bool isSerializable>
inline unsigned int AbstractProperty<
	DataType,
	isFundamental,
	isSerializable
>::getDimensions() const{
	return indexDescriptor.getDimensions();
}

template<typename DataType, bool isFundamental, bool isSerializable>
//inline const int* AbstractProperty<
inline std::vector<int> AbstractProperty<
	DataType,
	isFundamental,
	isSerializable
>::getRanges() const{
	return indexDescriptor.getRanges();
}

template<typename DataType, bool isFundamental, bool isSerializable>
inline int AbstractProperty<
	DataType,
	isFundamental,
	isSerializable
>::getOffset(
	const Index &index
) const{
	return blockSize*indexDescriptor.getLinearIndex(
		index,
		allowIndexOutOfBoundsAccess
	);
}

template<typename DataType, bool isFundamental, bool isSerializable>
inline const IndexDescriptor& AbstractProperty<
	DataType,
	isFundamental,
	isSerializable
>::getIndexDescriptor(
) const{
	return indexDescriptor;
}

template<typename DataType, bool isFundamental, bool isSerializable>
inline bool AbstractProperty<
	DataType,
	isFundamental,
	isSerializable
>::contains(
	const Index &index
) const{
	return indexDescriptor.contains(index);
}

template<typename DataType, bool isFundamental, bool isSerializable>
inline const DataType& AbstractProperty<
	DataType,
	isFundamental,
	isSerializable
>::operator()(
	const Index &index,
	unsigned int offset
) const{
//	return data[getOffset(index) + offset];
	int indexOffset = getOffset(index);
	if(indexOffset < 0)
		return defaultValue;
	else
		return data[indexOffset + offset];
}

template<typename DataType, bool isFundamental, bool isSerializable>
inline DataType& AbstractProperty<
	DataType,
	isFundamental,
	isSerializable
>::operator()(
	const Index &index,
	unsigned int offset
){
//	return data[getOffset(index) + offset];
	int indexOffset = getOffset(index);
	if(indexOffset < 0){
		static DataType defaultValueNonConst = defaultValue;
		return defaultValueNonConst;
	}
	else{
		return data[indexOffset + offset];
	}
}

template<typename DataType, bool isFundamental, bool isSerializable>
inline const DataType& AbstractProperty<
	DataType,
	isFundamental,
	isSerializable
>::operator()(unsigned int offset) const{
	return data[offset];
}

template<typename DataType, bool isFundamental, bool isSerializable>
inline DataType& AbstractProperty<
	DataType,
	isFundamental,
	isSerializable
>::operator()(unsigned int offset){
	return data[offset];
}

template<typename DataType, bool isFundamental, bool isSerializable>
inline void AbstractProperty<
	DataType,
	isFundamental,
	isSerializable
>::setAllowIndexOutOfBoundsAccess(bool allowIndexOutOfBoundsAccess){
	this->allowIndexOutOfBoundsAccess = allowIndexOutOfBoundsAccess;
}

template<typename DataType, bool isFundamental, bool isSerializable>
inline void AbstractProperty<
	DataType,
	isFundamental,
	isSerializable
>::setDefaultValue(const DataType &defaultValue){
	this->defaultValue = defaultValue;
}

template<>
inline std::string AbstractProperty<
	bool,
	true,
	false
>::serialize(Mode mode) const{
	switch(mode){
	case Mode::JSON:
	{
		nlohmann::json j;
		j["id"] = "AbstractProperty";
		j["indexDescriptor"] = nlohmann::json::parse(
			indexDescriptor.serialize(mode)
		);
		j["blockSize"] = blockSize;
		j["size"] = size;
		for(unsigned int n = 0; n < size; n++)
			j["data"].push_back(data[n]);

		return j.dump();
	}
	default:
		TBTKExit(
			"AbstractProperty<DataType>::serialize()",
			"Only Serializable::Mode::JSON is supported yet.",
			""
		);
	}
}

template<>
inline std::string AbstractProperty<
	char,
	true,
	false
>::serialize(Mode mode) const{
	switch(mode){
	case Mode::JSON:
	{
		nlohmann::json j;
		j["id"] = "AbstractProperty";
		j["indexDescriptor"] = nlohmann::json::parse(
			indexDescriptor.serialize(mode)
		);
		j["blockSize"] = blockSize;
		j["size"] = size;
		for(unsigned int n = 0; n < size; n++)
			j["data"].push_back(data[n]);

		return j.dump();
	}
	default:
		TBTKExit(
			"AbstractProperty<DataType>::serialize()",
			"Only Serializable::Mode::JSON is supported yet.",
			""
		);
	}
}

template<>
inline std::string AbstractProperty<
	int,
	true,
	false
>::serialize(Mode mode) const{
	switch(mode){
	case Mode::JSON:
	{
		nlohmann::json j;
		j["id"] = "AbstractProperty";
		j["indexDescriptor"] = nlohmann::json::parse(
			indexDescriptor.serialize(mode)
		);
		j["blockSize"] = blockSize;
		j["size"] = size;
		for(unsigned int n = 0; n < size; n++)
			j["data"].push_back(data[n]);

		return j.dump();
	}
	default:
		TBTKExit(
			"AbstractProperty<DataType>::serialize()",
			"Only Serializable::Mode::JSON is supported yet.",
			""
		);
	}
}

template<>
inline std::string AbstractProperty<
	float,
	true,
	false
>::serialize(Mode mode) const{
	switch(mode){
	case Mode::JSON:
	{
		nlohmann::json j;
		j["id"] = "AbstractProperty";
		j["indexDescriptor"] = nlohmann::json::parse(
			indexDescriptor.serialize(mode)
		);
		j["blockSize"] = blockSize;
		j["size"] = size;
		for(unsigned int n = 0; n < size; n++)
			j["data"].push_back(data[n]);

		return j.dump();
	}
	default:
		TBTKExit(
			"AbstractProperty<DataType>::serialize()",
			"Only Serializable::Mode::JSON is supported yet.",
			""
		);
	}
}

template<>
inline std::string AbstractProperty<
	double,
	true,
	false
>::serialize(Mode mode) const{
	switch(mode){
	case Mode::JSON:
	{
		nlohmann::json j;
		j["id"] = "AbstractProperty";
		j["indexDescriptor"] = nlohmann::json::parse(
			indexDescriptor.serialize(mode)
		);
		j["blockSize"] = blockSize;
		j["size"] = size;
		for(unsigned int n = 0; n < size; n++)
			j["data"].push_back(data[n]);

		return j.dump();
	}
	default:
		TBTKExit(
			"AbstractProperty<DataType>::serialize()",
			"Only Serializable::Mode::JSON is supported yet.",
			""
		);
	}
}

template<>
inline std::string AbstractProperty<std::complex<double>, false, false>::serialize(Mode mode) const{
	switch(mode){
	case Mode::JSON:
	{
		nlohmann::json j;
		j["id"] = "AbstractProperty";
		j["indexDescriptor"] = nlohmann::json::parse(
			indexDescriptor.serialize(mode)
		);
		j["blockSize"] = blockSize;
		j["size"] = size;
		for(unsigned int n = 0; n < size; n++){
//			std::stringstream ss;
//			ss << "(" << real(data[n]) << "," << imag(data[n]) << ")";
			std::string s = Serializable::serialize(data[n], mode);
			j["data"].push_back(s);
		}

		return j.dump();
	}
	default:
		TBTKExit(
			"AbstractProperty<DataType>::serialize()",
			"Only Serializable::Mode::JSON is supported yet.",
			""
		);
	}
}

template<>
inline std::string AbstractProperty<SpinMatrix, false, false>::serialize(Mode mode) const{
	TBTKNotYetImplemented("AbstractProperty<SpinMatrix, false, false>::serialize()");
}

template<typename DataType, bool isFundamental, bool isSerializable>
AbstractProperty<
	DataType,
	isFundamental,
	isSerializable
>::AbstractProperty() :
	indexDescriptor(IndexDescriptor::Format::None)
{
	this->blockSize = 0;

	size = blockSize;
	data = nullptr;
}

template<typename DataType, bool isFundamental, bool isSerializable>
AbstractProperty<
	DataType,
	isFundamental,
	isSerializable
>::AbstractProperty(
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

template<typename DataType, bool isFundamental, bool isSerializable>
AbstractProperty<
	DataType,
	isFundamental,
	isSerializable
>::AbstractProperty(
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

template<typename DataType, bool isFundamental, bool isSerializable>
AbstractProperty<
	DataType,
	isFundamental,
	isSerializable
>::AbstractProperty(
	unsigned int dimensions,
	const int *ranges,
	unsigned int blockSize
) :
	indexDescriptor(IndexDescriptor::Format::Ranges)
{
	this->blockSize = blockSize;

/*	indexDescriptor.setDimensions(dimensions);
	int *thisRanges = indexDescriptor.getRanges();
	for(unsigned int n = 0; n < dimensions; n++)
		thisRanges[n] = ranges[n];*/
	indexDescriptor.setRanges(ranges, dimensions);

	size = blockSize*indexDescriptor.getSize();
	data = new DataType[size];
	for(unsigned int n = 0; n < size; n++)
		data[n] = 0.;
}

template<typename DataType, bool isFundamental, bool isSerializable>
AbstractProperty<
	DataType,
	isFundamental,
	isSerializable
>::AbstractProperty(
	unsigned int dimensions,
	const int *ranges,
	unsigned int blockSize,
	const DataType *data
) :
	indexDescriptor(IndexDescriptor::Format::Ranges)
{
	this->blockSize = blockSize;

/*	indexDescriptor.setDimensions(dimensions);
	int *thisRanges = indexDescriptor.getRanges();
	for(unsigned int n = 0; n < dimensions; n++)
		thisRanges[n] = ranges[n];*/
	indexDescriptor.setRanges(ranges, dimensions);

	size = blockSize*indexDescriptor.getSize();
	this->data = new DataType[size];
	for(unsigned int n = 0; n < size; n++)
		this->data[n] = data[n];
}

template<typename DataType, bool isFundamental, bool isSerializable>
AbstractProperty<
	DataType,
	isFundamental,
	isSerializable
>::AbstractProperty(
	const IndexTree &indexTree,
	unsigned int blockSize
) :
	indexDescriptor(IndexDescriptor::Format::Custom)
{
	TBTKAssert(
		indexTree.getLinearMapIsGenerated(),
		"AbstractProperty::AbstractProperty()",
		"Linear map not constructed for the IndexTree.",
		"Call IndexTree::generateLinearIndex() before passing the"
		" IndexTree to the AbstractProperty constructor."
	);

	this->blockSize = blockSize;

	indexDescriptor.setIndexTree(indexTree);

	size = blockSize*indexDescriptor.getSize();
	data = new DataType[size];
	for(unsigned int n = 0; n < size; n++)
		data[n] = 0.;
}

template<typename DataType, bool isFundamental, bool isSerializable>
AbstractProperty<
	DataType,
	isFundamental,
	isSerializable
>::AbstractProperty(
	const IndexTree &indexTree,
	unsigned int blockSize,
	const DataType *data
) :
	indexDescriptor(IndexDescriptor::Format::Custom)
{
	TBTKAssert(
		indexTree.getLinearMapIsGenerated(),
		"AbstractProperty::AbstractProperty()",
		"Linear map not constructed for the IndexTree.",
		"Call IndexTree::generateLinearIndex() before passing the"
		" IndexTree to the AbstractProperty constructor."
	);

	this->blockSize = blockSize;

	indexDescriptor.setIndexTree(indexTree);

	size = blockSize*indexDescriptor.getSize();
	this->data = new DataType[size];
	for(unsigned int n = 0; n < size; n++)
		this->data[n] = data[n];
}

template<typename DataType, bool isFundamental, bool isSerializable>
AbstractProperty<
	DataType,
	isFundamental,
	isSerializable
>::AbstractProperty(
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

template<typename DataType, bool isFundamental, bool isSerializable>
AbstractProperty<
	DataType,
	isFundamental,
	isSerializable
>::AbstractProperty(
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

template<>
inline AbstractProperty<double, true, false>::AbstractProperty(
	const std::string &serialization,
	Mode mode
) :
	indexDescriptor(
		Serializable::extract(serialization, mode, "indexDescriptor"),
		mode
	)
{
	TBTKAssert(
		validate(serialization, "AbstractProperty", mode),
		"AbstractProperty::AbstractProperty()",
		"Unable to parse string as AbstractProperty '" << serialization
		<< "'.",
		""
	);

	switch(mode){
	case Mode::JSON:
		try{
			nlohmann::json j = nlohmann::json::parse(serialization);
			blockSize = j.at("blockSize").get<unsigned int>();
			size = j.at("size").get<unsigned int>();
			data = new double[size];
			nlohmann::json d = j.at("data");
			unsigned int counter = 0;
			for(
				nlohmann::json::iterator it = d.begin();
				it < d.end();
				++it
			){
				data[counter] = *it;
				counter++;
			}
		}
		catch(nlohmann::json::exception e){
			TBTKExit(
				"AbstractProperty::AbstractProperty()",
				"Unable to parse string as AbstractProperty '"
				<< serialization << "'.",
				""
			);
		}

		break;
	default:
		TBTKExit(
			"AbstractProperty::AbstractProperty()",
			"Only Serializable::Mode::JSON is supported yet.",
			""
		);
	}
}

template<>
inline AbstractProperty<std::complex<double>, false, false>::AbstractProperty(
	const std::string &serialization,
	Mode mode
) :
	indexDescriptor(
		Serializable::extract(serialization, mode, "indexDescriptor"),
		mode
	)
{
	TBTKAssert(
		validate(serialization, "AbstractProperty", mode),
		"AbstractProperty::AbstractProperty()",
		"Unable to parse string as AbstractProperty '" << serialization
		<< "'.",
		""
	);

	switch(mode){
	case Mode::JSON:
		try{
			nlohmann::json j = nlohmann::json::parse(serialization);
			blockSize = j.at("blockSize").get<unsigned int>();
			size = j.at("size").get<unsigned int>();
			data = new std::complex<double>[size];
			nlohmann::json d = j.at("data");
			unsigned int counter = 0;
			for(
				nlohmann::json::iterator it = d.begin();
				it < d.end();
				++it
			){
				std::complex<double> c;
				Serializable::deserialize(it->get<std::string>(), &c, mode);
				data[counter] = c;
				counter++;
			}
		}
		catch(nlohmann::json::exception e){
			TBTKExit(
				"AbstractProperty::AbstractProperty()",
				"Unable to parse string as AbstractProperty '"
				<< serialization << "'.",
				""
			);
		}

		break;
	default:
		TBTKExit(
			"AbstractProperty::AbstractProperty()",
			"Only Serializable::Mode::JSON is supported yet.",
			""
		);
	}
}

template<>
inline AbstractProperty<SpinMatrix, false, false>::AbstractProperty(
	const std::string &serialization,
	Mode mode
) :
	indexDescriptor(
		Serializable::extract(serialization, mode, "indexDescriptor"),
		mode
	)
{
	TBTKNotYetImplemented("AbstractProperty<SpinMatrix, false, false>::AbstractProperty()");
}

template<typename DataType, bool isFundamental, bool isSerializable>
inline AbstractProperty<
	DataType,
	isFundamental,
	isSerializable
>::~AbstractProperty(){
	if(data != nullptr)
		delete [] data;
}

template<typename DataType, bool isFundamental, bool isSerializable>
AbstractProperty<
	DataType,
	isFundamental,
	isSerializable
>& AbstractProperty<
	DataType,
	isFundamental,
	isSerializable
>::operator=(
	const AbstractProperty &rhs
){
	if(this != &rhs){
		indexDescriptor = rhs.indexDescriptor;

		blockSize = rhs.blockSize;

		size = rhs.size;
		if(data != nullptr)
			delete [] data;

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

template<typename DataType, bool isFundamental, bool isSerializable>
AbstractProperty<
	DataType,
	isFundamental,
	isSerializable
>& AbstractProperty<
	DataType,
	isFundamental,
	isSerializable
>::operator=(
	AbstractProperty &&rhs
){
	if(this != &rhs){
		indexDescriptor = std::move(rhs.indexDescriptor);

		blockSize = rhs.blockSize;

		size = rhs.size;
		if(data != nullptr)
			delete [] data;

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
