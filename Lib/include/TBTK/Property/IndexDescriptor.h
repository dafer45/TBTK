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
 *  @file IndexDescriptor.h
 *  @brief Describes the index structure of data stored for several indices.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_INDEX_DESCRIPTOR
#define COM_DAFER45_TBTK_INDEX_DESCRIPTOR

#include "TBTK/IndexTree.h"
#include "TBTK/Serializable.h"
#include "TBTK/TBTKMacros.h"

namespace TBTK{

/** @brief Describes the index structure of data stored for several indices. */
class IndexDescriptor : public Serializable{
public:
	/** Enum class determining the storage format. */
	enum class Format {None, Ranges, Custom};

	/** Constructor. */
	IndexDescriptor(Format format);

	/** Copy constructor. */
	IndexDescriptor(const IndexDescriptor &indexDescriptor);

	/** Move constructor. */
	IndexDescriptor(IndexDescriptor &&indexDescriptor);

	/** Constructor. Construct the IndexDescriptor from a serialization
	 *  string. */
	IndexDescriptor(const std::string &serialization, Mode mode);

	/** Destructor. */
	~IndexDescriptor();

	/** Assignment operator. */
	IndexDescriptor& operator=(const IndexDescriptor &rhs);

	/** Move assignment operator. */
	IndexDescriptor& operator=(IndexDescriptor &&rhs);

	/** Get format. */
	Format getFormat() const;

	/** Set dimensions. */
	void setDimensions(unsigned int dimensions);

	/** Get dimensions. */
	unsigned int getDimensions() const;

	/** Set ranges. */
	void setRanges(const int *ranges, unsigned int dimensions);

	/** Get ranges. */
//	int* getRanges();
	std::vector<int> getRanges() const;

	/** Get ranges. */
//	const int* getRanges() const;

	/** Set IndexTree. */
	void setIndexTree(const IndexTree &indexTree);

	/** Get IndexTree. */
	const IndexTree& getIndexTree() const;

	/** Get linear index. */
	int getLinearIndex(
		const Index &index,
		bool returnNegativeForMissingIndex = false
	) const;

	/** Get size. */
	unsigned int getSize() const;

	/** Returns true if the index descriptor contains the given index. */
	bool contains(const Index &index) const;

	/** Implements Serializable::serializea(). */
	virtual std::string serialize(Mode mode) const;
private:
	/** Index descriptor format. */
	Format format;

	class NoneFormat{
	public:
	};

	class RangeFormat{
	public:
		/** Number of dimensions. */
		unsigned int dimensions;

		/** Ranges. */
		int *ranges;
	};

	class CustomFormat{
	public:
		/** IndexTree. */
		IndexTree *indexTree;
	};

	/** Union of descriptor formats. */
	union Descriptor{
		NoneFormat noneFormat;
		RangeFormat rangeFormat;
		CustomFormat customFormat;
	};

	/** Actuall descriptor. */
	Descriptor descriptor;
};

inline IndexDescriptor::Format IndexDescriptor::getFormat() const{
	return format;
}

inline void IndexDescriptor::setDimensions(unsigned int dimensions){
	TBTKAssert(
		format == Format::Ranges,
		"IndexDescriptor::setDimensions()",
		"The IndexDescriptor is not of the format Format::Ranges.",
		""
	);
	descriptor.rangeFormat.dimensions = dimensions;
	if(descriptor.rangeFormat.ranges != NULL)
		delete [] descriptor.rangeFormat.ranges;
	descriptor.rangeFormat.ranges = new int[dimensions];
}

inline unsigned int IndexDescriptor::getDimensions() const{
	TBTKAssert(
		format == Format::Ranges,
		"IndexDescriptor::getDimensions()",
		"The IndexDescriptor is not of the format Format::Ranges.",
		""
	);
	return descriptor.rangeFormat.dimensions;
}

inline void IndexDescriptor::setRanges(const int *ranges, unsigned int dimensions){
	TBTKAssert(
		format == Format::Ranges,
		"IndexDescriptor::setRanges()",
		"The IndexDescriptor is not of the format Format::Ranges.",
		""
	);
	descriptor.rangeFormat.dimensions = dimensions;
	if(descriptor.rangeFormat.ranges != NULL)
		delete [] descriptor.rangeFormat.ranges;
	descriptor.rangeFormat.ranges = new int[dimensions];
	for(unsigned int n = 0; n < dimensions; n++)
		descriptor.rangeFormat.ranges[n] = ranges[n];
}

//inline int* IndexDescriptor::getRanges(){
inline std::vector<int> IndexDescriptor::getRanges() const{
	TBTKAssert(
		format == Format::Ranges,
		"IndexDescriptor::setDimensions()",
		"The IndexDescriptor is not of the format Format::Ranges.",
		""
	);
//	return descriptor.rangeFormat.ranges;

	std::vector<int> ranges;
	for(unsigned int n = 0; n < descriptor.rangeFormat.dimensions; n++)
		ranges.push_back(descriptor.rangeFormat.ranges[n]);

	return ranges;
}

/*inline const int* IndexDescriptor::getRanges() const{
	TBTKAssert(
		format == Format::Ranges,
		"IndexDescriptor::setDimensions()",
		"The IndexDescriptor is not of the format Format::Ranges.",
		""
	);
	return descriptor.rangeFormat.ranges;
}*/

inline const IndexTree& IndexDescriptor::getIndexTree() const{
	TBTKAssert(
		format == Format::Custom,
		"IndexDescriptor::getIndexTree()",
		"The IndexDescriptor is not of the format Format::Custom.",
		""
	);

	return *descriptor.customFormat.indexTree;
}

inline int IndexDescriptor::getLinearIndex(
	const Index &index,
	bool returnNegativeForMissingIndex
) const{
	TBTKAssert(
		format == Format::Custom,
		"IndexDescriptor::getOffset()",
		"The IndexDescriptor is not of the format Format::Custom.",
		""
	);

	return descriptor.customFormat.indexTree->getLinearIndex(
		index,
		IndexTree::SearchMode::MatchWildcards,
		returnNegativeForMissingIndex
	);
}

inline bool IndexDescriptor::contains(const Index &index) const{
	TBTKAssert(
		format == Format::Custom,
		"IndexDescriptor::contains()",
		"The IndexDescriptor is not of the format Format::Custom.",
		""
	);

	if(
		descriptor.customFormat.indexTree->getLinearIndex(
			index,
			IndexTree::SearchMode::StrictMatch,
			true
		) == -1
	){
		return false;
	}
	else{
		return true;
	}
}

};	//End namespace TBTK

#endif
