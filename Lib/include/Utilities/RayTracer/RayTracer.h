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

#include "Density.h"
#include "Magnetization.h"
#include "Model.h"
#include "WaveFunction.h"
#include "TBTKMacros.h"
#include "Vector3d.h"

#include <functional>
#include <initializer_list>

namespace TBTK{

class RayTracer{
public:
	/** COnstructor. */
	RayTracer();

	/** Destructor. */
	~RayTracer();

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
		const Property::WaveFunction &waveFunction,
		unsigned int state
	);
private:
	/** Class for encoding RGB colors. */
	class Color{
	public:
		double r, g, b;
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
	private:
		/** Render context. */
		const RenderContext *renderContext;

		/** Ray direction. */
		Vector3d rayDirection;

		/** Index that was hit. */
		Index index;

		/** Coordinate. */
		Vector3d coordinate;

		/** Direction from object. */
		Vector3d *directionFromObject;
	};

	/** Perform ray tracing. */
	void trace(
		const IndexDescriptor &indexDescriptor,
		const Model &model,
		std::function<Color(HitDescriptor &hitDescriptor)> &&lambdaColorPicker
	);
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

inline void RayTracer::setUp(std::initializer_list<double> up){
	renderContext.setUp(up);
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

};	//End namespace TBTK

#endif
