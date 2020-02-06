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
 *  @file Serializable.h
 *  @brief Abstract base class for serializable objects.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_SERIALIZABLE
#define COM_DAFER45_TBTK_SERIALIZABLE

#include "TBTK/SpinMatrix.h"
#include "TBTK/Statistics.h"
#include "TBTK/TBTKMacros.h"

#include <complex>
#include <sstream>
#include <type_traits>
#include <vector>

namespace TBTK{

//Forward declaration of classes that are pseudo-Serializable (that implements
//the Serializable interface non-virtually).
class Index;
class HoppingAmplitude;
template<typename DataType> class CArray;

class Serializable{
public:
	/** Serialization modes. Note that debug is not guaranteed to be
	 *  backward compatible. */
	enum class Mode {Debug, Binary, XML, JSON};

	/** Serialize object. */
	virtual std::string serialize(Mode mode) const = 0;

	/** Returns true if the serialization string has an ID. */
	static bool hasID(const std::string &serialization, Mode mode);

	/** Get the ID of a serialization string. */
	static std::string getID(const std::string &serialization, Mode mode);

	/** Extract a component from a container serialization. When
	 *  serializing classes, their parent are embedded inside the child
	 *  class serialization. I.e. the parent class is a component contained
	 *  in the child class serialization. However, when reconstructing a
	 *  serialized class, its parent constructor has to be called before
	 *  the child class is constructed. This function provides the means
	 *  to extract the serialization of a parent class from the
	 *  serialization of a child class.
	 *
	 *  @param containerID ID for the container (child class).
	 *  @param componentID ID for the component (parent class).
	 *  @param componentName Name used to refer to the parent object. Only
	 *  used in some modes.
	 *
	 *  @param mode Mode with which the string has been serialized.
	 *
	 *  @return The serialization string for the component. */
	static std::string extractComponent(
		const std::string &serialization,
		const std::string &containerID,
		const std::string &componentID,
		const std::string &componentName,
		Mode mode
	);
protected:
	/** Validate serialization string. */
	static bool validate(
		const std::string &serialization,
		const std::string &id,
		Mode mode
	);

	/** Get the content of a serializtion string. */
	static std::string getContent(
		const std::string &serialization,
		Mode mode
	);

	/** Split content string. */
	static std::vector<std::string> split(
		const std::string &content,
		Mode mode
	);

	/** Serialize.
	 *
	 *  @param data The data to serialize.
	 *  @param mode The mode with which to serialize the data.
	 *
	 *  @return A string containing the serialization of the data. */
	template<typename DataType>
	static typename std::enable_if<!std::is_pointer<DataType>::value, std::string>::type serialize(const DataType &data, Mode mode);
	template<typename DataType>
	static typename std::enable_if<std::is_pointer<DataType>::value, std::string>::type serialize(const DataType &data, Mode mode);

	/** Deserialize.
	 *
	 *  @param serialization The serialization from which the data is to be
	 *  deserialized.
	 *
	 *  @param mode The mode with which the data has been serialized.
	 *
	 *  @return The data after deserialization. */
	template<typename DataType>
	static DataType deserialize(
		const std::string &serialization,
		Mode mode
	);
private:
	/** Serialize bool. */
	static std::string _serialize(bool b, Mode mode);

	/** Deserialize bool. */
	static void _deserialize(
		const std::string &serialization,
		bool *b,
		Mode mode
	);

	/** Serialize int. */
	static std::string _serialize(int i, Mode mode);

	/** Deserialize int. */
	static void _deserialize(
		const std::string &serialization,
		int *i,
		Mode mode
	);

	/** Serialize unsigned int. */
	static std::string _serialize(unsigned int u, Mode mode);

	/** Deserialize unsigned int. */
	static void _deserialize(
		const std::string &serialization,
		unsigned int *u,
		Mode mode
	);

	/** Serialize double. */
	static std::string _serialize(double d, Mode mode);

	/** Deserialize double. */
	static void _deserialize(
		const std::string &serialization,
		double *d,
		Mode mode
	);

	/** Serialize complex<double>. */
	static std::string _serialize(std::complex<double> c, Mode mode);

	/** Deserialize complex<double>. */
	static void _deserialize(
		const std::string &serialization,
		std::complex<double> *c,
		Mode mode
	);

	/** Serialize Statistics. */
	static std::string _serialize(Statistics s, Mode mode);

	/** Deserialize Statistics. */
	static void _deserialize(
		const std::string &serialization,
		Statistics *s,
		Mode mode
	);
protected:
	/** Extracts a component of a serialization string, catching potential
	 *  errors. In particular used to extract the serialization string for
	 *  parents in the constructor of a child, because no try-catch block
	 *  is possible to insert at this point. */
	static std::string extract(
		const std::string &serialization,
		Mode mode,
		std::string component
	);

	/** Friend classes (Classes that are psudo-Serializable because they
	 *  are so small and often used that a virtual function would have a
	 *  non-negligible performance penalty). */
	friend class Index;
	friend class HoppingAmplitude;
	friend class SourceAmplitude;
	friend class OverlapAmplitude;
	template<typename DataType> friend class CArray;
};

//#ifndef TBTK_DISABLE_NLOHMANN_JSON
template<typename DataType>
typename std::enable_if<!std::is_pointer<DataType>::value, std::string>::type Serializable::serialize(const DataType &data, Mode mode){
	return data.serialize(mode);
}

template<typename DataType>
typename std::enable_if<std::is_pointer<DataType>::value, std::string>::type Serializable::serialize(const DataType &data, Mode mode){
	return data->serialize(mode);
}
/*template<typename DataType>
typename std::enable_if<true, std::string>::type Serializable::serialize(const DataType &data, Mode mode){
	return data->serialize(mode);
}*/
//#endif

template<>
inline std::string Serializable::serialize(const bool &data, Mode mode){
	return _serialize(data, mode);
}

template<>
inline std::string Serializable::serialize(const double &data, Mode mode){
	return _serialize(data, mode);
}

template<>
inline std::string Serializable::serialize(
	const std::complex<double> &data,
	Mode mode
){
	return _serialize(data, mode);
}

template<>
inline std::string Serializable::serialize(const int &data, Mode mode){
	return _serialize(data, mode);
}

template<>
inline std::string Serializable::serialize(const unsigned int &data, Mode mode){
	return _serialize(data, mode);
}

template<>
inline std::string Serializable::serialize(const SpinMatrix &data, Mode mode){
	TBTKNotYetImplemented("Serializable::serialize<SpinMatrix>()");
}

template<>
inline std::string Serializable::serialize(const Statistics &data, Mode mode){
	return _serialize(data, mode);
}

template<>
inline std::string Serializable::serialize(
	const std::vector<std::complex<double>> &data,
	Mode mode
){
	TBTKNotYetImplemented(
		"Serializable::serialize<std::vector<std::complex<double>>>()"
	);
}

template<typename DataType>
DataType Serializable::deserialize(const std::string &serialization, Mode mode){
	return DataType(serialization, mode);
}

template<>
inline int Serializable::deserialize(const std::string &serialization, Mode mode){
	int i;
	_deserialize(serialization, &i, mode);
	return i;
}

template<>
inline unsigned int Serializable::deserialize(const std::string &serialization, Mode mode){
	unsigned int i;
	_deserialize(serialization, &i, mode);
	return i;
}

template<>
inline double Serializable::deserialize(const std::string &serialization, Mode mode){
	double d;
	_deserialize(serialization, &d, mode);
	return d;
}

template<>
inline std::complex<double> Serializable::deserialize(
	const std::string &serialization,
	Mode mode
){
	std::complex<double> c;
	_deserialize(serialization, &c, mode);
	return c;
}

template<>
inline SpinMatrix Serializable::deserialize(
	const std::string &serialization,
	Mode mode
){
	TBTKNotYetImplemented("Serializable::deserialize<SpinMatrix>()");
}

template<>
inline Statistics Serializable::deserialize(
	const std::string &serialization,
	Mode mode
){
	Statistics s;
	_deserialize(serialization, &s, mode);
	return s;
}

template<>
inline std::vector<std::complex<double>> Serializable::deserialize(
	const std::string &serialization,
	Mode mode
){
	TBTKNotYetImplemented(
		"Serializable::deserialize<std::vector<std::complex<double>>>()"
	);
}

inline std::string Serializable::_serialize(bool b, Mode mode){
	switch(mode){
	case Mode::Debug:
	case Mode::JSON:
	{
		std::stringstream ss;
		ss << b;

		return ss.str();
	}
	default:
		TBTKExit(
			"Serializable::serialize()",
			"Only Mode::Debug is supported yet.",
			""
		);
	}
}

inline void Serializable::_deserialize(
	const std::string &serialization,
	bool *b,
	Mode mode
){
	switch(mode){
	case Mode::Debug:
	case Mode::JSON:
	{
		std::stringstream ss;
		ss.str(serialization);
		ss >> *b;

		break;
	}
	default:
		TBTKExit(
			"Serializable::deserialize()",
			"Only Mode::Debug is supported yet.",
			""
		);
	}
}

inline std::string Serializable::_serialize(int i, Mode mode){
	switch(mode){
	case Mode::Debug:
	case Mode::JSON:
	{
		std::stringstream ss;
		ss << i;

		return ss.str();
	}
	default:
		TBTKExit(
			"Serializable::serialize()",
			"Only Mode::Debug is supported yet.",
			""
		);
	}
}

inline void Serializable::_deserialize(
	const std::string &serialization,
	int *i,
	Mode mode
){
	switch(mode){
	case Mode::Debug:
	case Mode::JSON:
	{
		std::stringstream ss;
		ss.str(serialization);
		ss >> *i;

		break;
	}
	default:
		TBTKExit(
			"Serializable::deserialize()",
			"Only Mode::Debug is supported yet.",
			""
		);
	}
}

inline std::string Serializable::_serialize(unsigned int u, Mode mode){
	switch(mode){
	case Mode::Debug:
	case Mode::JSON:
	{
		std::stringstream ss;
		ss << u;

		return ss.str();
	}
	default:
		TBTKExit(
			"Serializable::serialize()",
			"Only Mode::Debug is supported yet.",
			""
		);
	}
}

inline void Serializable::_deserialize(
	const std::string &serialization,
	unsigned int *u,
	Mode mode
){
	switch(mode){
	case Mode::Debug:
	case Mode::JSON:
	{
		std::stringstream ss;
		ss.str(serialization);
		ss >> *u;

		break;
	}
	default:
		TBTKExit(
			"Serializable::deserialize()",
			"Only Mode::Debug is supported yet.",
			""
		);
	}
}

inline std::string Serializable::_serialize(double d, Mode mode){
	switch(mode){
	case Mode::Debug:
	case Mode::JSON:
	{
		std::stringstream ss;
		ss << d;

		return ss.str();
	}
	default:
		TBTKExit(
			"Serializable::serialize()",
			"Only Mode::Debug is supported yet.",
			""
		);
	}
}

inline void Serializable::_deserialize(
	const std::string &serialization,
	double *d,
	Mode mode
){
	switch(mode){
	case Mode::Debug:
	case Mode::JSON:
	{
		std::stringstream ss;
		ss.str(serialization);
		ss >> *d;

		break;
	}
	default:
		TBTKExit(
			"Serializable::deserialize()",
			"Only Mode::Debug is supported yet.",
			""
		);
	}
}

inline std::string Serializable::_serialize(std::complex<double> c, Mode mode){
	switch(mode){
	case Mode::Debug:
	case Mode::JSON:
	{
		std::stringstream ss;
		ss << c;

		return ss.str();
	}
	default:
		TBTKExit(
			"Serializable::serialize()",
			"Only Mode::Debug is supported yet.",
			""
		);
	}
}

inline void Serializable::_deserialize(
	const std::string &serialization,
	std::complex<double> *c,
	Mode mode
){
	switch(mode){
	case Mode::Debug:
	case Mode::JSON:
	{
		std::stringstream ss;
		ss.str(serialization);
		ss >> *c;

		break;
	}
	default:
		TBTKExit(
			"Serializable::deserialize()",
			"Only Mode::Debug is supported yet.",
			""
		);
	}
}

inline std::string Serializable::_serialize(Statistics statistics, Mode mode){
	switch(mode){
	case Mode::Debug:
	{
		std::stringstream ss;
		switch(statistics){
		case Statistics::FermiDirac:
		case Statistics::BoseEinstein:
			ss << static_cast<int>(statistics);
			break;
		default:
			TBTKExit(
				"Serializable::serialize()",
				"Unknown Statistics type '" << static_cast<int>(statistics) << "'",
				"This should never happen, contact the developer."
			);
		}

		return ss.str();
	}
	case Mode::JSON:
	{
		std::stringstream ss;
		switch(statistics){
		case Statistics::FermiDirac:
			ss << "FermiDirac";
			break;
		case Statistics::BoseEinstein:
			ss << "BoseEinstein";
			break;
		default:
			TBTKExit(
				"Serializable::serialize()",
				"Unknown Statistics type '" << static_cast<int>(statistics) << "'",
				"This should never happen, contact the developer."
			);
		}

		return ss.str();
	}
	default:
		TBTKExit(
			"Serializable::serialize()",
			"Only Mode::Debug is supported yet.",
			""
		);
	}
}

inline void Serializable::_deserialize(
	const std::string &serialization,
	Statistics *statistics,
	Mode mode
){
	switch(mode){
	case Mode::Debug:
	{
		std::stringstream ss;
		ss.str(serialization);
		int i;
		ss >> i;
		switch(i){
		case static_cast<int>(Statistics::FermiDirac):
			*statistics = Statistics::FermiDirac;
			break;
		case static_cast<int>(Statistics::BoseEinstein):
			*statistics = Statistics::BoseEinstein;
			break;
		default:
			TBTKExit(
				"Serializable::serialize()",
				"Unknown Statistics type '" << i << "'",
				"The serialization string is either corrupted"
				<< " or the the serialization was created with"
				<< " a newer version of TBTK that supports"
				<< " more types of Statistics."
			);
		}

		break;
	}
	case Mode::JSON:
	{
		if(serialization.compare("FermiDirac") == 0){
			*statistics = Statistics::FermiDirac;
		}
		else if(serialization.compare("BoseEinstein") == 0){
			*statistics = Statistics::BoseEinstein;
		}
		else{
			TBTKExit(
				"Serializable::serialize()",
				"Unknown Statistics type '" << serialization
				<< "'",
				"The serialization string is either corrupted"
				<< " or the the serialization was created with"
				<< " a newer version of TBTK that supports"
				<< " more types of Statistics."
			);
		}

		break;
	}
	default:
		TBTKExit(
			"Serializable::deserialize()",
			"Only Mode::Debug is supported yet.",
			""
		);
	}
}

};	//End namespace TBTK

#endif
