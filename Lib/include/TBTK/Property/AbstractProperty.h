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

#include "TBTK/Property/Property.h"
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
class AbstractProperty : public Property, public Serializable{
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
	const std::vector<DataType>& getData() const;

	/** Get data. Same as AbstractProperty::getData(), but with write
	 *  access.
	 *
	 *  @return The data on the raw format described in the detailed
	 *  description. */
	std::vector<DataType>& getDataRW();

	/** Get the dimension of the data. [Only works for the Ranges format.]
	 *
	 *  @return The dimension of the grid that the data is calculated on.
	 */
	unsigned int getDimensions() const;

	/** Get the ranges for the dimensions of the density. [Only works for
	 *  the Ranges format.]
	 *
	 *  @return A list of ranges for the different dimensions of the grid
	 *  the data is calculated on. */
	std::vector<int> getRanges() const;

	/** Get the memory offset that should be used to access the raw data
	 *  for the given Index. [Only works for the Custom format.]
	 *
	 *  @param index Index for which to get the offset.
	 *
	 *  @return The memory offset for the given Index. */
	int getOffset(const Index &index) const;

	/** Get IndexDescriptor.
	 *
	 *  @return The IndexDescriptor that is used internally to handle the
	 *  different formats.*/
	const IndexDescriptor& getIndexDescriptor() const;

	/** Returns true if the property contains data for the given index.
	 *  [Only works for the Custom format.]
	 *
	 *  @param index Index to check.
	 *
	 *  @raturn True if the Property contains data for the given Index. */
	bool contains(const Index &index) const;

	/** Function call operator. Returns the data element for the given
	 *  Index and offset. By default the Property does not accept @link
	 *  Index Indices @endlink that are not contained in the Property.
	 *  However, AbstractProperty::setAllowIndexOutOfBounds(true) is called
	 *  the Property will return the value set by a call to
	 *  AbstractProperty::setDefaultVlue(). [Only works for the Custom
	 *  format.]
	 *
	 *  @param index The Index to get the data for.
	 *  @param offset The offset to apply inside the block for the given
	 *  Index. If not specified, the first element is returned. The
	 *  function can therefore be used without specifying the offset when
	 *  Properties with only a single element per block. [Only works for
	 *  the Custom format.]
	 *
	 *  @return The data element for the given Index and offset. */
	virtual const DataType& operator()(
		const Index &index,
		unsigned int offset = 0
	) const;

	/** Function call operator. Returns the data element for the given
	 *  Index and offset. By default the Property does not accept @link
	 *  Index Indices @endlink that are not contained in the Property.
	 *  However, AbstractProperty::setAllowIndexOutOfBounds(true) is called
	 *  the Property will return the value set by a call to
	 *  AbstractProperty::setDefaultVlue(). Note that although it is safe
	 *  to write to out of bounds elements, doing so does not result in the
	 *  value being stored in the Property. [Only works for Format::Custom].
	 *
	 *  @param index The Index to get the data for.
	 *  @param offset The offset to apply inside the block for the given
	 *  Index. If not specified, the first element is returned. The
	 *  function can therefore be used without specifying the offset when
	 *  Properties with only a single element per block. [Only works for
	 *  the Custom format.]
	 *
	 *  @return The data element for the given Index and offset. */
	virtual DataType& operator()(
		const Index &index,
		unsigned int offset = 0
	);

	/** Alias for operator()(const Index &index, unsigned int offset = 0).
	 *  Ensures that operator()(unsigned int offset) is not called when
	 *  calling the function operator default offset and a single subindex
	 *  index operator()({1}). Without this {1} would be cast to 1 instead
	 *  of Index({1}).
	 *
	 *  @param index The Index to get the data for.
	 *
	 *  @return The data element for the given Index. */
	DataType& operator()(const std::initializer_list<int> &index);

	/** Alias for operator()(const Index &index, unsigned int offset = 0).
	 *  Ensures that operator()(unsigned int offset) is not called when
	 *  calling the function operator default offset and a single subindex
	 *  index operator()({1}). Without this {1} would be cast to 1 instead
	 *  of Index({1}).
	 *
	 *  @param index The Index to get the data for.
	 *
	 *  @return The data element for the given Index. */
	const DataType& operator()(
		const std::initializer_list<int> &index
	) const;

	/** Alias for operator()(const Index &index, unsigned int offset = 0).
	 *  Ensures that operator()(unsigned int offset) is not called when
	 *  calling the function operator default offset and a single subindex
	 *  index operator()({1}). Without this {1} would be cast to 1 instead
	 *  of Index({1}).
	 *
	 *  @param index The Index to get the data for.
	 *
	 *  @return The data element for the given Index. */
	DataType& operator()(const std::initializer_list<unsigned int> &index);

	/** Alias for operator()(const Index &index, unsigned int offset = 0).
	 *  Ensures that operator()(unsigned int offset) is not called when
	 *  calling the function operator default offset and a single subindex
	 *  index operator()({1}). Without this {1} would be cast to 1 instead
	 *  of Index({1}).
	 *
	 *  @param index The Index to get the data for.
	 *
	 *  @return The data element for the given Index. */
	const DataType& operator()(
		const std::initializer_list<unsigned int> &index
	) const;

	/** Function call operator. Returns the data element for the given
	 *  offset.
	 *
	 *  @param offset The offset to apply. The offset is an index into the
	 *  raw data.
	 *
	 *  @return The data element for the given offset. */
	virtual const DataType& operator()(unsigned int offset) const;

	/** Function call operator. Returns the data element for the given
	 *  offset.
	 *
	 *  @param offset The offset to apply. The offset is an index into the
	 *  raw data.
	 *
	 *  @return The data element for the given offset. */
	virtual DataType& operator()(unsigned int offset);

	/** Set whether the access of data elements for @link Index Indices
	 *  @endlink that are not contained in the Property is allowed or not
	 *  when using the function operator AbstractProperty::operator(). If
	 *  enabled, remember to also initialize the value that is used for out
	 *  of bounds access using AbstractProperty::setDefaultValue(). [Only
	 *  meaningful for the Custom format.]
	 *
	 *  @param allowIndexOutOfBoundsAccess True to enable out of bounds
	 *  access. */
	void setAllowIndexOutOfBoundsAccess(bool allowIndexOutOfBoundsAccess);

	/** Set the value that is returned when accessing indices not contained
	 *  in the Property using the function operator
	 *  AbstractProperty::operator(). Only used if
	 *  AbstractProperty::setAllowIndexOutOfBoundsAccess(true) is called.
	 *  [Only meaningful for the Custom format.]
	 *
	 *  @param defaultValue The value that will be returned for out of
	 *  bounds access.*/
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
	 *  @param ranges A list of upper limits for the ranges of the
	 *  different dimensions. The nth dimension will have the range [0,
	 *  ranges[n]).
	 *
	 *  @param blockSize The number of data elements per block. */
	AbstractProperty(
		const std::vector<int> &ranges,
		unsigned int blockSize
	);

	/** Constructs an AbstractProperty with the Ranges format and fills the
	 *  internal memory with the data provided as input.
	 *
	 *  @param ranges A list of upper limits for the ranges of the
	 *  different dimensions. The nth dimension will have the range [0,
	 *  ranges[n]).
	 *
	 *  @param blockSize The number of data elements per block.
	 *  @param data The data stored on the raw format described in the
	 *  detailed description of the class. */
	AbstractProperty(
		const std::vector<int> &ranges,
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

	/** Addition assignment operator. Child classes that want to use this
	 *  function must override this function and call the parents addition
	 *  assignment operator after having performed checks whether the
	 *  addition is possible, and before performing the addition of the
	 *  child class parameters.
	 *
	 *  @param The AbstractProperty to be added to the left hand side.
	 *
	 *  @return The left hand side after the right hand side has been
	 *  added. */
	AbstractProperty& operator+=(const AbstractProperty &rhs);

	/** Subtraction assignment operator. Child classes that want to use
	 *  this function must override this function and call the parents
	 *  subtraction assignment operator after having performed checks
	 *  whether the subtraction is possible, and before performing the
	 *  subtraction of the child class parameters.
	 *
	 *  @param The AbstractProperty to be subtracted from the left hand
	 *  side.
	 *
	 *  @return The left hand side after the right hand side has been
	 *  subtracted. */
	AbstractProperty& operator-=(const AbstractProperty &rhs);

	/** Multiplication assignment operator. Child classes that want to use
	 *  this function must override this function and call the parents
	 *  multiplication assignment operator after having performed checks
	 *  whether the multiplication is possible, and before performing the
	 *  multiplication of the child class parameters.
	 *
	 *  @param The value to multiply the left hand side by.
	 *
	 *  @return The left hand side after having been multiplied by the
	 *  right hand side. */
	AbstractProperty& operator*=(const DataType &rhs);

	/** Division assignment operator. Child classes that want to use this
	 *  function must override this function and call the parents division
	 *  assignment operator after having performed checks whether the
	 *  division is possible, and before performing the division of the
	 *  child class parameters.
	 *
	 *  @param The value to divide the left hand side by.
	 *
	 *  @return The left hand side after having been divided by the right
	 *  hand side. */
	AbstractProperty& operator/=(const DataType &rhs);
private:
	/** IndexDescriptor describing the memory layout of the data. */
	IndexDescriptor indexDescriptor;

	/** Size of a single block of data needed to describe a single point
	 *  refered to by an index. In particular:
	 *  indexDescriptor.size()*blockSize = size. */
	unsigned int blockSize;

	/** Data. */
	std::vector<DataType> data;

	/** Flag indicating whether access of */
	bool allowIndexOutOfBoundsAccess;

	/** Default value used for out of bounds access. */
	DataType defaultValue;

	/** Function to be overloaded by Properties that support dynamic
	 *  calculation. The overloading function should calculate the property
	 *  for the given Index and write it into the provided block. If no
	 *  overload is provided, the AbstractProperty defaults to print an
	 *  error message telling that dynamic calculation of the Property is
	 *  not possible.
	 *
	 *  @param index The Index for which the Property should be calculated.
	 *  @param block Pointer to the first element of the block into which
	 *  the Property should be written. */
	virtual void calculateDynamically(const Index &index, DataType *block);
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
	return data.size();
}

template<typename DataType, bool isFundamental, bool isSerializable>
inline const std::vector<DataType>& AbstractProperty<
	DataType,
	isFundamental,
	isSerializable
>::getData() const{
	return data;
}

template<typename DataType, bool isFundamental, bool isSerializable>
inline std::vector<DataType>& AbstractProperty<
	DataType,
	isFundamental,
	isSerializable
>::getDataRW(){
	return data;
}

template<typename DataType, bool isFundamental, bool isSerializable>
inline unsigned int AbstractProperty<
	DataType,
	isFundamental,
	isSerializable
>::getDimensions() const{
	return indexDescriptor.getRanges().size();
}

template<typename DataType, bool isFundamental, bool isSerializable>
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
	int indexOffset = getOffset(index);
	if(indexOffset < 0){
		static DataType defaultValueNonConst;
		defaultValueNonConst = defaultValue;
		return defaultValueNonConst;
	}
	else{
		return data[indexOffset + offset];
	}
}

template<typename DataType, bool isFundamental, bool isSerializeable>
inline DataType& AbstractProperty<
	DataType,
	isFundamental,
	isSerializeable
>::operator()(const std::initializer_list<int> &index){
	return operator()(index, 0);
}

template<typename DataType, bool isFundamental, bool isSerializeable>
inline const DataType& AbstractProperty<
	DataType,
	isFundamental,
	isSerializeable
>::operator()(const std::initializer_list<int> &index) const{
	return operator()(index, 0);
}

template<typename DataType, bool isFundamental, bool isSerializeable>
inline DataType& AbstractProperty<
	DataType,
	isFundamental,
	isSerializeable
>::operator()(const std::initializer_list<unsigned int> &index){
	return operator()(index, 0);
}

template<typename DataType, bool isFundamental, bool isSerializeable>
inline const DataType& AbstractProperty<
	DataType,
	isFundamental,
	isSerializeable
>::operator()(const std::initializer_list<unsigned int> &index) const{
	return operator()(index, 0);
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
		for(unsigned int n = 0; n < data.size(); n++){
			//Convert the reference data[n] to an actual bool to
			//get the code to compile on Mac. Some issue with the
			//nlohmann library on Mac. Replace by the single
			//commented out line when it is working again.
			bool d = data[n];
			j["data"].push_back(d);
//			j["data"].push_back(data[n]);
		}

		j["allowIndexOutOfBoundsAccess"] = allowIndexOutOfBoundsAccess;
		j["defaultValue"] = defaultValue;

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
		for(unsigned int n = 0; n < data.size(); n++)
			j["data"].push_back(data[n]);

		j["allowIndexOutOfBoundsAccess"] = allowIndexOutOfBoundsAccess;
		j["defaultValue"] = defaultValue;

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
		for(unsigned int n = 0; n < data.size(); n++)
			j["data"].push_back(data[n]);

		j["allowIndexOutOfBoundsAccess"] = allowIndexOutOfBoundsAccess;
		j["defaultValue"] = defaultValue;

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
		for(unsigned int n = 0; n < data.size(); n++)
			j["data"].push_back(data[n]);

		j["allowIndexOutOfBoundsAccess"] = allowIndexOutOfBoundsAccess;
		j["defaultValue"] = defaultValue;

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
		for(unsigned int n = 0; n < data.size(); n++)
			j["data"].push_back(data[n]);

		j["allowIndexOutOfBoundsAccess"] = allowIndexOutOfBoundsAccess;
		j["defaultValue"] = defaultValue;

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
		for(unsigned int n = 0; n < data.size(); n++){
			std::string s = Serializable::serialize(data[n], mode);
			j["data"].push_back(s);
		}

		j["allowIndexOutOfBoundsAccess"] = allowIndexOutOfBoundsAccess;
		j["defaultValue"] = Serializable::serialize(defaultValue, mode);

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
	indexDescriptor()
{
	this->blockSize = 0;

	allowIndexOutOfBoundsAccess = false;
}

template<typename DataType, bool isFundamental, bool isSerializable>
AbstractProperty<
	DataType,
	isFundamental,
	isSerializable
>::AbstractProperty(
	unsigned int blockSize
) :
	indexDescriptor()
{
	this->blockSize = blockSize;

	unsigned int size = blockSize;
	data.reserve(size);
	for(unsigned int n = 0; n < size; n++)
		data.push_back(0.);

	allowIndexOutOfBoundsAccess = false;
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
	indexDescriptor()
{
	this->blockSize = blockSize;

	unsigned int size = blockSize;
	this->data.reserve(size);
	for(unsigned int n = 0; n < size; n++)
		this->data.push_back(data[n]);

	allowIndexOutOfBoundsAccess = false;
}

template<typename DataType, bool isFundamental, bool isSerializable>
AbstractProperty<
	DataType,
	isFundamental,
	isSerializable
>::AbstractProperty(
	const std::vector<int> &ranges,
	unsigned int blockSize
) :
	indexDescriptor(ranges)
{
	this->blockSize = blockSize;

	unsigned int size = blockSize*indexDescriptor.getSize();
	data.reserve(size);
	for(unsigned int n = 0; n < size; n++)
		data.push_back(DataType(0.));

	allowIndexOutOfBoundsAccess = false;
}

template<typename DataType, bool isFundamental, bool isSerializable>
AbstractProperty<
	DataType,
	isFundamental,
	isSerializable
>::AbstractProperty(
	const std::vector<int> &ranges,
	unsigned int blockSize,
	const DataType *data
) :
	indexDescriptor(ranges)
{
	this->blockSize = blockSize;

	unsigned int size = blockSize*indexDescriptor.getSize();
	this->data.reserve(size);
	for(unsigned int n = 0; n < size; n++)
		this->data.push_back(data[n]);

	allowIndexOutOfBoundsAccess = false;
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
	indexDescriptor(indexTree)
{
	TBTKAssert(
		indexTree.getLinearMapIsGenerated(),
		"AbstractProperty::AbstractProperty()",
		"Linear map not constructed for the IndexTree.",
		"Call IndexTree::generateLinearIndex() before passing the"
		" IndexTree to the AbstractProperty constructor."
	);

	this->blockSize = blockSize;

	unsigned int size = blockSize*indexDescriptor.getSize();
	data.reserve(size);
	for(unsigned int n = 0; n < size; n++)
		data.push_back(DataType(0.));

	allowIndexOutOfBoundsAccess = false;
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
	indexDescriptor(indexTree)
{
	TBTKAssert(
		indexTree.getLinearMapIsGenerated(),
		"AbstractProperty::AbstractProperty()",
		"Linear map not constructed for the IndexTree.",
		"Call IndexTree::generateLinearIndex() before passing the"
		" IndexTree to the AbstractProperty constructor."
	);

	this->blockSize = blockSize;

	unsigned int size = blockSize*indexDescriptor.getSize();
	this->data.reserve(size);
	for(unsigned int n = 0; n < size; n++)
		this->data.push_back(data[n]);

	allowIndexOutOfBoundsAccess = false;
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

	data = abstractProperty.data;

	allowIndexOutOfBoundsAccess
		= abstractProperty.allowIndexOutOfBoundsAccess;
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

	data = abstractProperty.data;

	allowIndexOutOfBoundsAccess
		= abstractProperty.allowIndexOutOfBoundsAccess;
}

template<>
inline AbstractProperty<int, true, false>::AbstractProperty(
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
			nlohmann::json d = j.at("data");
			for(
				nlohmann::json::iterator it = d.begin();
				it < d.end();
				++it
			){
				data.push_back(*it);
			}

			allowIndexOutOfBoundsAccess = j.at(
				"allowIndexOutOfBoundsAccess"
			).get<bool>();
			defaultValue = j.at("defaultValue").get<double>();
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
			nlohmann::json d = j.at("data");
			for(
				nlohmann::json::iterator it = d.begin();
				it < d.end();
				++it
			){
				data.push_back(*it);
			}

			allowIndexOutOfBoundsAccess = j.at(
				"allowIndexOutOfBoundsAccess"
			).get<bool>();
			defaultValue = j.at("defaultValue").get<double>();
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
			nlohmann::json d = j.at("data");
			for(
				nlohmann::json::iterator it = d.begin();
				it < d.end();
				++it
			){
				std::complex<double> c;
				Serializable::deserialize(it->get<std::string>(), &c, mode);
				data.push_back(c);
			}

			allowIndexOutOfBoundsAccess = j.at(
				"allowIndexOutOfBoundsAccess"
			).get<bool>();
			Serializable::deserialize(
				j.at("defaultValue").get<std::string>(),
				&defaultValue,
				mode
			);
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

		data = rhs.data;

		allowIndexOutOfBoundsAccess = rhs.allowIndexOutOfBoundsAccess;
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

		data = rhs.data;

		allowIndexOutOfBoundsAccess = rhs.allowIndexOutOfBoundsAccess;
	}

	return *this;
}

template<typename DataType, bool isFundamental, bool isSerializable>
inline AbstractProperty<DataType, isFundamental, isSerializable>&
AbstractProperty<
	DataType,
	isFundamental,
	isSerializable
>::operator+=(
	const AbstractProperty<DataType, isFundamental, isSerializable> &rhs
){
	TBTKAssert(
		indexDescriptor == rhs.indexDescriptor,
		"AbstractProperty::operator+=()",
		"Incompatible Properties. The Properties does not have the"
		<< " same index structure.",
		""
	);

	TBTKAssert(
		blockSize == rhs.blockSize,
		"AbstractProperty::operator+=()",
		"Incompatible Properties. The Properties does not have the"
		<< " same block size.",
		""
	);

	TBTKAssert(
		allowIndexOutOfBoundsAccess == rhs.allowIndexOutOfBoundsAccess,
		"AbstractProperty::operator+=()",
		"Incompatible Properties. The Properties differ in their"
		<< " 'index out of bounds behavior'.",
		"Use AbstractProperty::setAllowIndexOutOfBoundsAccess() to set"
		<< " the 'index out of bounds' behavior."
	);

	if(allowIndexOutOfBoundsAccess){
		TBTKAssert(
			defaultValue == rhs.defaultValue,
			"AbstractProperty::operator+=()",
			"Incompatible Properties. The Properties differ in"
			<< " their default values.",
			"Use AbstractProperty::setDefaultValue() to set the"
			<< " default value."
		);
	}

	TBTKAssert(
		data.size() == rhs.data.size(),
		"AbstractProperty::operator+=()",
		"Incompatible Properties. The Properties have different data"
		<< " sizes.",
		"This should never happen, contact the developer."
	);

	for(unsigned int n = 0; n < data.size(); n++)
		data[n] += rhs.data[n];

	return *this;
}

template<typename DataType, bool isFundamental, bool isSerializable>
inline AbstractProperty<DataType, isFundamental, isSerializable>&
AbstractProperty<
	DataType,
	isFundamental,
	isSerializable
>::operator-=(
	const AbstractProperty<DataType, isFundamental, isSerializable> &rhs
){
	TBTKAssert(
		indexDescriptor == rhs.indexDescriptor,
		"AbstractProperty::operator-=()",
		"Incompatible Properties. The Properties does not have the"
		<< " same index structure.",
		""
	);

	TBTKAssert(
		blockSize == rhs.blockSize,
		"AbstractProperty::operator-=()",
		"Incompatible Properties. The Properties does not have the"
		<< " same block size.",
		""
	);

	TBTKAssert(
		allowIndexOutOfBoundsAccess == rhs.allowIndexOutOfBoundsAccess,
		"AbstractProperty::operator-=()",
		"Incompatible Properties. The Properties differ in their"
		<< " 'index out of bounds behavior'.",
		"Use AbstractProperty::setAllowIndexOutOfBoundsAccess() to set"
		<< " the 'index out of bounds' behavior."
	);

	if(allowIndexOutOfBoundsAccess){
		TBTKAssert(
			defaultValue == rhs.defaultValue,
			"AbstractProperty::operator-=()",
			"Incompatible Properties. The Properties differ in"
			<< " their default values.",
			"Use AbstractProperty::setDefaultValue() to set the"
			<< " default value."
		);
	}

	TBTKAssert(
		data.size() == rhs.data.size(),
		"AbstractProperty::operator-=()",
		"Incompatible Properties. The Properties have different data"
		<< " sizes.",
		"This should never happen, contact the developer."
	);

	for(unsigned int n = 0; n < data.size(); n++)
		data[n] -= rhs.data[n];

	return *this;
}

template<typename DataType, bool isFundamental, bool isSerializable>
inline AbstractProperty<DataType, isFundamental, isSerializable>&
AbstractProperty<
	DataType,
	isFundamental,
	isSerializable
>::operator*=(
	const DataType &rhs
){
	if(allowIndexOutOfBoundsAccess)
		defaultValue *= rhs;

	for(unsigned int n = 0; n < data.size(); n++)
		data[n] *= rhs;

	return *this;
}

template<typename DataType, bool isFundamental, bool isSerializable>
inline AbstractProperty<DataType, isFundamental, isSerializable>&
AbstractProperty<
	DataType,
	isFundamental,
	isSerializable
>::operator/=(
	const DataType &rhs
){
	if(allowIndexOutOfBoundsAccess)
		defaultValue /= rhs;

	for(unsigned int n = 0; n < data.size(); n++)
		data[n] /= rhs;

	return *this;
}

template<typename DataType, bool isFundamental, bool isSerializable>
void AbstractProperty<
	DataType,
	isFundamental,
	isSerializable
>::calculateDynamically(const Index &index, DataType *block){
	TBTKExit(
		"AbstractProperty::calculateDynamically()",
		"This Property does not support dynamic calculation.",
		""
	);
}

};	//End namespace Property
};	//End namespace TBTK

#endif
