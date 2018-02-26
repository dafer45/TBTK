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
 *  @file RayTracer.h
 *  @brief Creates figures of properties using ray tracing.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_RAY_TRACER
#define COM_DAFER45_TBTK_RAY_TRACER

#include "FieldWrapper.h"
#include "Model.h"
#include "Property/Density.h"
#include "Property/LDOS.h"
#include "Property/Magnetization.h"
#include "Property/WaveFunctions.h"
#include "TBTKMacros.h"
#include "Vector3d.h"

#include <functional>
#include <initializer_list>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace TBTK{

class RayTracer{
public:
	/** Constructor. */
	RayTracer();

	/** Copy constructor. */
	RayTracer(const RayTracer &rayTracer);

	/** Move constructor. */
	RayTracer(RayTracer &&rayTracer);

	/** Destructor. */
	~RayTracer();

	/** Assignment operator. */
	RayTracer& operator=(const RayTracer &rhs);

	/** Move assignment operator. */
	RayTracer& operator=(RayTracer &&rhs);

	/** Set camera position. */
	void setCameraPosition(const Vector3d &cameraPosition);

	/** Set camera position. */
	void setCameraPosition(std::initializer_list<double> cameraPosition);

	/** Set camera focus. */
	void setFocus(const Vector3d &focus);

	/** Set camera focus. */
	void setFocus(std::initializer_list<double> focus);

	/** Set up direction. */
	void setUp(const Vector3d &up);

	/** Set up direction. */
	void setUp(std::initializer_list<double> up);

	/** Set viewport width. */
	void setWidth(unsigned int width);

	/** Set viewport height. */
	void setHeight(unsigned int height);

	/** Set state radius. */
	void setStateRadius(double stateRadius);

	/** Set number of deflections. */
	void setNumDeflections(unsigned int numDeflections);

	/** Get number of deflections. */
	unsigned int getNumDeflections() const;

	/** Set ray length. */
	void setRayLength(double rayLength);

	/** Get ray length. */
	double getRayLength() const;

	/** Set number of ray segments. */
	void setNumRaySegments(unsigned int numRaySegments);

	/** Get number of ray segments. */
	unsigned int getNumRaySegments() const;

	/** Plot Density. */
	void plot(const Model& model, const Property::Density &density);

	/** Plot Magnetization. */
	void plot(
		const Model &model,
		const Property::Magnetization &magnetization
	);

	/** Plot Magnetization. */
	void plot(
		const Model &model,
		const Property::WaveFunctions &waveFunctions,
		unsigned int state
	);

	/** Plot a field. */
	void plot(
		Field<std::complex<double>, double> &field
	);

	/** Plot fields. */
	void plot(
		const std::vector<const FieldWrapper*> &fields
	);

	/** Interactive. */
	void interactivePlot(
		const Model &model,
		const Property::LDOS &ldos,
		double sigma = 0,
		unsigned int windowSize = 51
	);

	/** Save result to file. */
	void save(std::string filename);
private:
	/** Class for encoding RGB colors. */
	class Color{
	public:
		double r, g, b;
	};

	/** Class for describing materials. */
	class Material{
	public:
		/** Constructor. */
		Material();

		/** Color. */
		Color color;

		/** Light properties. */
		double ambient, diffusive, emissive, specular;

		/** Default material parameters. */
		static constexpr double DEFAULT_AMBIENT = 1;
		static constexpr double DEFAULT_DIFFUSIVE = 0.5;
		static constexpr double DEFAULT_EMISSIVE = 0;
		static constexpr double DEFAULT_SPECULAR = 0.1;
	};

	/** */
	class RenderContext{
	public:
		/** Constructor. */
		RenderContext();

		/** Destructor. */
		~RenderContext();

		/** Set camera position. */
		void setCameraPosition(const Vector3d &cameraPosition);

		/** Set camera position. */
		void setCameraPosition(std::initializer_list<double> cameraPosition);

		/** Get cameraPosition. */
		const Vector3d& getCameraPosition() const;

		/** Set focus. */
		void setFocus(const Vector3d &focus);

		/** Set camera focus. */
		void setFocus(std::initializer_list<double> focus);

		/** Get focus point. */
		const Vector3d& getFocus() const;

		/** Set up direction. */
		void setUp(const Vector3d &up);

		/** Set up direction. */
		void setUp(std::initializer_list<double> up);

		/** Get up direction. */
		const Vector3d& getUp() const;

		/** Set viewport width. */
		void setWidth(unsigned int width);

		/** Get viewport width. */
		unsigned int getWidth() const;

		/** Set viewport height. */
		void setHeight(unsigned int height);

		/** Get viewport height. */
		unsigned int getHeight() const;

		/** Set state radius. */
		void setStateRadius(double stateRadius);

		/** Get state radius. */
		double getStateRadius() const;

		/** Set number of deflections. */
		void setNumDeflections(unsigned int numDeflections);

		/*** Get number of deflections. */
		unsigned int getNumDeflections() const;

		/** Set ray length. */
		void setRayLength(double rayLength);

		/** Get ray length. */
		double getRayLength() const;

		/** Set number of ray segments. */
		void setNumRaySegments(unsigned int numRaySegments);

		/** Get number of ray segments. */
		unsigned int getNumRaySegments() const;
	private:
		/** Camera position. */
		Vector3d cameraPosition;

		/** Focus point. */
		Vector3d focus;

		/** Up direction. */
		Vector3d up;

		/** Viewport width. */
		double width;

		/** Viewport height. */
		double height;

		/** State radius. */
		double stateRadius;

		/** Maximum number of times a ray will be traced after having
		 *  been deflected. */
		unsigned int numDeflections;

		/** Ray length used to trace fields. */
		double rayLength;

		/** Number of ray segments used when tracing fields. */
		unsigned int numRaySegments;
	};

	RenderContext renderContext;

	/** */
	class HitDescriptor{
	public:
		/** Constructor. */
		HitDescriptor(const RenderContext &renderContext);

		/** Copy construtor. */
		HitDescriptor(const HitDescriptor &hitDescriptor);

		/** Move constructor. */
		HitDescriptor(HitDescriptor &&hitDescriptor);

		/** Destructor. */
		~HitDescriptor();

		/** Assignment operator. */
		HitDescriptor& operator=(const HitDescriptor &rhs);

		/** Move assignment operator. */
		HitDescriptor& operator=(HitDescriptor &&rhs);

		/** Set ray source. */
		void setRaySource(const Vector3d &raySource);

		/** Get ray source. */
		const Vector3d& getRaySource() const;

		/** Set ray direction. */
		void setRayDirection(const Vector3d &rayDirection);

		/** Get ray direction. */
		const Vector3d& getRayDirection() const;

		/** Set index. */
		void setIndex(const Index &index);

		/** Get index. */
		const Index& getIndex() const;

		/** Set coordinate. */
		void setCoordinate(const Vector3d coordainte);

		/** Get coordinate. */
		const Vector3d& getCoordinate() const;

		/** Get directionFromObject. */
		const Vector3d& getDirectionFromObject();

		/** Get impact position. */
		const Vector3d& getImpactPosition();
	private:
		/** Render context. */
		const RenderContext *renderContext;

		/** Ray source. */
		Vector3d raySource;

		/** Ray direction. */
		Vector3d rayDirection;

		/** Index that was hit. */
		Index index;

		/** Coordinate. */
		Vector3d coordinate;

		/** Direction from object. */
		Vector3d *directionFromObject;

		/** Impact position. */
		Vector3d *impactPosition;
	};

	class RenderResult{
	public:
		/** Constructor. */
		RenderResult(unsigned int width, unsigned int height);

		/** Copy constructor. */
		RenderResult(const RenderResult &renderResult);

		/** Move constructor. */
		RenderResult(RenderResult &&renderResult);

		/** Destructor. */
		~RenderResult();

		/** Assignment operator. */
		RenderResult& operator=(const RenderResult &rhs);

		/** Move assignment operator. */
		RenderResult& operator=(RenderResult &&rhs);

		/** Get canvas. */
		cv::Mat& getCanvas();

		/** Get canvas. */
		const cv::Mat& getCanvas() const;

		/** Get HitDescriptors. */
		std::vector<HitDescriptor>** getHitDescriptors();
	private:
		/** Width and height of the canvas and hitDescriptors array. */
		unsigned int width, height;

		/** Canvas. */
		cv::Mat canvas;

		/** Array of HitDescriptors containing information about all the sites
		 *  that were hit on each pixels during ray tracing.*/
		std::vector<HitDescriptor> **hitDescriptors;
	};

	RenderResult *renderResult;

	/** Run rendering procedure. */
	void render(
		const IndexDescriptor &indexDescriptor,
		const Model &model,
		const std::vector<const FieldWrapper*> &fields,
		std::function<Material(HitDescriptor &hitDescriptor)> &&lambdaColorPicker,
		std::function<void(cv::Mat &canvas, const Index &index)> &&lambdaInteractive = {}
	);

	/** Trace a ray. */
	Color trace(
		const std::vector<Vector3d> &coordinates,
		const Vector3d &raySource,
		const Vector3d &rayDirection,
		const IndexTree &indexTree,
		const std::vector<const FieldWrapper*> &fields,
		std::vector<HitDescriptor> &hitDescriptors,
		std::function<Material(HitDescriptor &hitDescriptor)> lambdaColorPicker,
		unsigned int deflections = 0
	);

	/** Trace a ray trough a set of fields. */
	Color traceFields(
		const std::vector<const FieldWrapper*> &fields,
		const Vector3d &raySource,
		const Vector3d &rayDirection
	);

	/** Returns the canvas converted to char-type. */
	cv::Mat getCharImage() const;

	/** Event handler for the interactive mode. */
	class EventHandler{
	public:
		/** Try to lock EventHandler. Returns true if successful. */
		static bool lock(
			RayTracer *owner,
			std::function<void(
				int event,
				int x,
				int y,
				int flags,
				void *userData
			)> &&lambdaOnMouseChange
		);

		/** Unlock EventHandler. */
		static bool unlock(const RayTracer *owner);

		/** On mouse change callback. */
		static void onMouseChange(
			int event,
			int x,
			int y,
			int flags,
			void *userdata
		);
	private:
		/** Flag indicating whether the EventHandler is locked. */
		static bool isLocked;

		/** Owner of the lock. */
		static RayTracer *owner;

		/** On mouse change lambda function. */
		static std::function<void(int event, int x, int y, int flags, void *userData)> &&lambdaOnMouseChange;
	};
};

inline void RayTracer::setCameraPosition(const Vector3d &cameraPosition){
	renderContext.setCameraPosition(cameraPosition);
}

inline void RayTracer::setCameraPosition(
	std::initializer_list<double> cameraPosition
){
	renderContext.setCameraPosition(cameraPosition);
}

inline void RayTracer::setFocus(const Vector3d &focus){
	renderContext.setFocus(focus);
}

inline void RayTracer::setFocus(std::initializer_list<double> focus){
	renderContext.setFocus(focus);
}

inline void RayTracer::setUp(const Vector3d &up){
	renderContext.setUp(up);
}

inline void RayTracer::setWidth(unsigned int width){
	renderContext.setWidth(width);
}

inline void RayTracer::setHeight(unsigned int height){
	renderContext.setHeight(height);
}

inline void RayTracer::setUp(std::initializer_list<double> up){
	renderContext.setUp(up);
}

inline void RayTracer::setStateRadius(double stateRadius){
	renderContext.setStateRadius(stateRadius);
}

inline void RayTracer::setNumDeflections(unsigned int numDeflections){
	renderContext.setNumDeflections(numDeflections);
}

inline unsigned int RayTracer::getNumDeflections() const{
	return renderContext.getNumDeflections();
}

inline void RayTracer::setRayLength(double rayLength){
	renderContext.setRayLength(rayLength);
}

inline double RayTracer::getRayLength() const{
	return renderContext.getRayLength();
}

inline void RayTracer::setNumRaySegments(unsigned int numRaySegments){
	renderContext.setNumRaySegments(numRaySegments);
}

inline unsigned int RayTracer::getNumRaySegments() const{
	return renderContext.getNumRaySegments();
}

/*inline RayTracer::RenderResult::RenderResult(
	unsigned int width,
	unsigned int height
){
	this->width = width;
	this->height = height;

	canvas = cv::Mat(height, width, CV_32FC3);

	hitDescriptors = new std::vector<HitDescriptor>*[width];
	for(unsigned int x = 0; x < width; x++)
		hitDescriptors[x] = new std::vector<HitDescriptor>[height];
}

inline RayTracer::RenderResult::~RenderResult(){
	for(unsigned int x = 0; x < width; x++)
		delete [] hitDescriptors[x];
	delete [] hitDescriptors;
}*/

inline cv::Mat& RayTracer::RenderResult::getCanvas(){
	return canvas;
}

inline const cv::Mat& RayTracer::RenderResult::getCanvas() const{
	return canvas;
}

inline std::vector<RayTracer::HitDescriptor>** RayTracer::RenderResult::getHitDescriptors(){
	return hitDescriptors;
}

inline RayTracer::Material::Material(){
	color.r = 0;
	color.g = 0;
	color.b = 0;
	ambient = DEFAULT_AMBIENT;
	diffusive = DEFAULT_AMBIENT;
	emissive = DEFAULT_EMISSIVE;
	specular = DEFAULT_SPECULAR;
}

inline void RayTracer::RenderContext::setCameraPosition(
	const Vector3d &cameraPosition
){
	this->cameraPosition = cameraPosition;
}

inline void RayTracer::RenderContext::setCameraPosition(
	std::initializer_list<double> cameraPosition
){
	TBTKAssert(
		cameraPosition.size() == 3,
		"RayTracer::setCameraPosition()",
		"Camera position can only have three coordinates.",
		""
	);

	this->cameraPosition.x = *(cameraPosition.begin() + 0);
	this->cameraPosition.y = *(cameraPosition.begin() + 1);
	this->cameraPosition.z = *(cameraPosition.begin() + 2);
}

inline const Vector3d& RayTracer::RenderContext::getCameraPosition() const{
	return cameraPosition;
}

inline void RayTracer::RenderContext::setFocus(const Vector3d &focus){
	this->focus = focus;
}

inline void RayTracer::RenderContext::setFocus(
	std::initializer_list<double> focus
){
	TBTKAssert(
		focus.size() == 3,
		"RayTracer::setFocus()",
		"Focus can only have three coordinates.",
		""
	);

	this->focus.x = *(focus.begin() + 0);
	this->focus.y = *(focus.begin() + 1);
	this->focus.z = *(focus.begin() + 2);
}

inline const Vector3d& RayTracer::RenderContext::getFocus() const{
	return focus;
}

inline void RayTracer::RenderContext::setUp(const Vector3d &up){
	this->up = up;
}

inline void RayTracer::RenderContext::setUp(
	std::initializer_list<double> up
){
	TBTKAssert(
		up.size() == 3,
		"RayTracer::setCameraPosition()",
		"Camera position can only have three coordinates.",
		""
	);

	this->up.x = *(up.begin() + 0);
	this->up.y = *(up.begin() + 1);
	this->up.z = *(up.begin() + 2);
}

inline const Vector3d& RayTracer::RenderContext::getUp() const{
	return up;
}

inline void RayTracer::RenderContext::setWidth(unsigned int width){
	this->width = width;
}

inline unsigned int RayTracer::RenderContext::getWidth() const{
	return width;
}

inline void RayTracer::RenderContext::setHeight(unsigned int height){
	this->height = height;
}

inline unsigned int RayTracer::RenderContext::getHeight() const{
	return height;
}

inline void RayTracer::RenderContext::setStateRadius(double stateRadius){
	this->stateRadius = stateRadius;
}

inline double RayTracer::RenderContext::getStateRadius() const{
	return stateRadius;
}

inline void RayTracer::RenderContext::setNumDeflections(
	unsigned int numDeflections
){
	this->numDeflections = numDeflections;
}

inline unsigned int RayTracer::RenderContext::getNumDeflections() const{
	return numDeflections;
}

inline void RayTracer::RenderContext::setRayLength(double rayLength){
	this->rayLength = rayLength;
}

inline double RayTracer::RenderContext::getRayLength() const{
	return rayLength;
}

inline void RayTracer::RenderContext::setNumRaySegments(
	unsigned int numRaySegments
){
	this->numRaySegments = numRaySegments;
}

inline unsigned int RayTracer::RenderContext::getNumRaySegments() const{
	return numRaySegments;
}

inline void RayTracer::HitDescriptor::setRaySource(
	const Vector3d &raySource
){
	this->raySource = raySource;
}

inline const Vector3d& RayTracer::HitDescriptor::getRaySource() const{
	return this->raySource;
}

inline void RayTracer::HitDescriptor::setRayDirection(
	const Vector3d &rayDirection
){
	this->rayDirection = rayDirection;
}

inline const Vector3d& RayTracer::HitDescriptor::getRayDirection() const{
	return this->rayDirection;
}

inline void RayTracer::HitDescriptor::setIndex(const Index &index){
	this->index = index;
}

inline const Index& RayTracer::HitDescriptor::getIndex() const{
	return index;
}

inline void RayTracer::HitDescriptor::setCoordinate(const Vector3d coordinate){
	this->coordinate = coordinate;
}

inline const Vector3d& RayTracer::HitDescriptor::getCoordinate() const{
	return coordinate;
}

inline bool RayTracer::EventHandler::lock(
	RayTracer *owner,
	std::function<void(
		int event,
		int x,
		int y,
		int flags,
		void *userData
	)> &&lambdaOnMouseChange
){
	if(isLocked){
		return false;
	}
	else{
		isLocked = true;
		EventHandler::owner = owner;
		EventHandler::lambdaOnMouseChange = lambdaOnMouseChange;
		return true;
	}
}

inline bool RayTracer::EventHandler::unlock(const RayTracer *owner){
	if(EventHandler::owner == owner){
		EventHandler::owner = nullptr;
		EventHandler::isLocked = false;
		EventHandler::lambdaOnMouseChange = nullptr;
		return true;
	}
	else{
		return false;
	}
}

inline void RayTracer::EventHandler::onMouseChange(
	int event,
	int x,
	int y,
	int flags,
	void *userData
){
	if(lambdaOnMouseChange){
		lambdaOnMouseChange(event, x, y, flags, userData);
	}
	else{
		TBTKExit(
			"RayTracer::EventHandler::onMouseChange()",
			"lambdaOnMouseChange is nullptr.",
			"This should never happen, contact the developer."
		);
	}
}

};	//End namespace TBTK

#endif
