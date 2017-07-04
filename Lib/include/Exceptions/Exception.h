#ifndef COM_DAFER45_TBTK_EXCEPTION
#define COM_DAFER45_TBTK_EXCEPTION

#include <exception>
#include <string>

namespace TBTK{

class Exception : public std::exception{
public:
	/** Constructor. */
	Exception(
		const std::string& function,
		const std::string& where,
		const std::string& message,
		const std::string& hint
	);

	/** Destructor. */
	virtual ~Exception();

	/** Overrider std::exception::what(). */
	virtual const char* what() const noexcept;

	/** Print exception to standard error output. */
	virtual void print() const;

	/** Get function. */
	const std::string& getFunction() const;

	/** Get where. */
	const std::string& getWhere() const;

	/** Get message. */
	const std::string& getMessage() const;

	/** Get hint. */
	const std::string& getHint() const;
private:
	/** Function throwing the exception. */
	std::string function;

	/** File name and line where the exception happened. */
	std::string where;

	/** Message describing what happened. */
	std::string message;

	/** Hint for how to resolve the problem. */
	std::string hint;
};

inline const std::string& Exception::getFunction() const{
	return function;
}

inline const std::string& Exception::getWhere() const{
	return where;
}

inline const std::string& Exception::getMessage() const{
	return message;
}

inline const std::string& Exception::getHint() const{
	return hint;
}

};	//End of namespace TBTK

#endif
