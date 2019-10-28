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

/** @file RayTracer.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Plot/Plotter.h"
#include "TBTK/RayTracer.h"
#include "TBTK/Smooth.h"
#include "TBTK/Streams.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

namespace TBTK{

bool RayTracer::EventHandler::isLocked = false;
RayTracer *RayTracer::EventHandler::owner = nullptr;
function<void(int event, int x, int y, int flags, void *userData)> &&RayTracer::EventHandler::lambdaOnMouseChange = {};

RayTracer::RayTracer(){
	renderResult = nullptr;
}

RayTracer::RayTracer(
	const RayTracer &rayTracer
) :
	renderContext(rayTracer.renderContext)
{
	if(rayTracer.renderResult == nullptr)
		renderResult = nullptr;
	else
		renderResult = new RenderResult(*rayTracer.renderResult);
}

RayTracer::RayTracer(
	RayTracer &&rayTracer
) :
	renderContext(std::move(rayTracer.renderContext))
{
	if(rayTracer.renderResult == nullptr){
		renderResult = nullptr;
	}
	else{
		renderResult = rayTracer.renderResult;
		rayTracer.renderResult = nullptr;
	}
}

RayTracer::~RayTracer(){
	if(renderResult != nullptr)
		delete renderResult;
}

RayTracer& RayTracer::operator=(const RayTracer &rhs){
	if(this != &rhs){
		renderContext = rhs.renderContext;
		if(rhs.renderResult == nullptr)
			renderResult = nullptr;
		else
			renderResult = new RenderResult(*rhs.renderResult);
	}

	return *this;
}

RayTracer& RayTracer::operator=(RayTracer &&rhs){
	if(this != &rhs){
		renderContext = std::move(rhs.renderContext);
		if(rhs.renderResult == nullptr){
			renderResult = nullptr;
		}
		else{
			renderResult = rhs.renderResult;
			rhs.renderResult = nullptr;
		}
	}

	return *this;
}

void RayTracer::plot(const Model& model, const Property::Density &density){
	const IndexDescriptor &indexDescriptor = density.getIndexDescriptor();
	TBTKAssert(
		indexDescriptor.getFormat() == IndexDescriptor::Format::Custom,
		"RayTracer::plot()",
		"Only storage format IndexDescriptor::Format::Custom supported.",
		"Use calculateProperty(patterns) instead of "
			<< "calculateProperty(pattern, ranges) when extracting"
			<< " properties."
	);

	double min = density.getMin();
	double max = density.getMax();

	vector<const FieldWrapper*> emptyFields;
	render(
		indexDescriptor,
		model,
		emptyFields,
		[&density, min, max](const HitDescriptor &hitDescriptor) -> RayTracer::Material
		{
			Material material;
			material.color.r = 255*(density(hitDescriptor.getIndex()) - min)/(max - min);
			material.color.g = 255*(density(hitDescriptor.getIndex()) - min)/(max - min);
			material.color.b = 255*(density(hitDescriptor.getIndex()) - min)/(max - min);

			return material;
		}
	);
}

void RayTracer::plot(
	const Model& model,
	const Property::Magnetization &magnetization
){
	const IndexDescriptor &indexDescriptor = magnetization.getIndexDescriptor();
	TBTKAssert(
		indexDescriptor.getFormat() == IndexDescriptor::Format::Custom,
		"RayTracer::plot()",
		"Only storage format IndexDescriptor::Format::Custom supported.",
		"Use calculateProperty(patterns) instead of "
			<< "calculateProperty(pattern, ranges) when extracting"
			<< " properties."
	);

	vector<const FieldWrapper*> emptyFields;
	render(
		indexDescriptor,
		model,
		emptyFields,
		[&magnetization](HitDescriptor &hitDescriptor) -> RayTracer::Material
		{
			Vector3d directionFromObject = hitDescriptor.getDirectionFromObject();
			const SpinMatrix& spinMatrix = magnetization(
				hitDescriptor.getIndex()
			);
			Vector3d spinVector = spinMatrix.getSpinVector();
			double projection = Vector3d::dotProduct(
				directionFromObject,
				spinVector
			);
//			double density = spinMatrix.getDensity();

			Material material;
			if(projection > 0){
				material.color.r = 255;
				material.color.g = 0;
				material.color.b = 0;
			}
			else{
				material.color.r = 255;
				material.color.g = 255;
				material.color.b = 255;
			}

			return material;
		}
	);
}

void RayTracer::plot(
	const Model& model,
	const Property::WaveFunctions &waveFunctions,
	unsigned int state
){
	const IndexDescriptor &indexDescriptor = waveFunctions.getIndexDescriptor();
	TBTKAssert(
		indexDescriptor.getFormat() == IndexDescriptor::Format::Custom,
		"RayTracer::plot()",
		"Only storage format IndexDescriptor::Format::Custom supported.",
		"Use calculateProperty(patterns) instead of "
			<< "calculateProperty(pattern, ranges) when extracting"
			<< " properties."
	);

//	double minAbs = waveFunction.getMinAbs();
	double maxAbs = waveFunctions.getMaxAbs();

	vector<const FieldWrapper*> emptyFields;
	render(
		indexDescriptor,
		model,
		emptyFields,
		[&waveFunctions, state, maxAbs](HitDescriptor &hitDescriptor) -> RayTracer::Material
		{
			complex<double> amplitude = waveFunctions(hitDescriptor.getIndex(), state);
			double absolute = abs(amplitude);
			absolute = absolute/maxAbs;
			double argument = arg(amplitude);
			if(argument < 0)
				argument += 2*M_PI;

			Material material;
			material.color.r = 255*absolute*(2*M_PI - argument)/(2*M_PI);
			material.color.g = 0;
			material.color.b = 255*absolute*argument/(2*M_PI);

			return material;
		}
	);
}

void RayTracer::plot(
	Field<complex<double>, double> &field
){
	vector<const FieldWrapper*> fields;
	fields.push_back(new FieldWrapper(field));

	//Dummies
	IndexTree emptyIndexTree;
	emptyIndexTree.generateLinearMap();
	IndexDescriptor emptyIndexDescriptor(emptyIndexTree);
	Model emptyModel;
	emptyModel.construct();

	render(
		emptyIndexDescriptor,
		emptyModel,
		fields,
		[](HitDescriptor &hitDescriptor) -> RayTracer::Material
		{
			return Material();
		}
	);
}

void RayTracer::plot(
	const vector<const FieldWrapper*> &fields
){
/*	vector<const FieldWrapper*> fields;
	for(unsigned int n
	fields.push_back(new FieldWrapper(field));*/

	//Dummies
	IndexTree emptyIndexTree;
	emptyIndexTree.generateLinearMap();
	IndexDescriptor emptyIndexDescriptor(emptyIndexTree);
	Model emptyModel;
	emptyModel.construct();

	render(
		emptyIndexDescriptor,
		emptyModel,
		fields,
		[](HitDescriptor &hitDescriptor) -> RayTracer::Material
		{
			return Material();
		}
	);
}

void RayTracer::interactivePlot(
	const Model &model,
	const Property::LDOS &ldos,
	double sigma,
	unsigned int windowSize
){
	const IndexDescriptor &indexDescriptor = ldos.getIndexDescriptor();
	TBTKAssert(
		indexDescriptor.getFormat() == IndexDescriptor::Format::Custom,
		"RayTracer::plot()",
		"Only storage format IndexDescriptor::Format::Custom supported.",
		"Use calculateProperty(patterns) instead of "
			<< "calculateProperty(pattern, ranges) when extracting"
			<< " properties."
	);

	vector<const FieldWrapper*> emptyFields;
	render(
		indexDescriptor,
		model,
		emptyFields,
		[&ldos](HitDescriptor &hitDescriptor) -> RayTracer::Material
		{
			Material material;
			material.color.r = 255;
			material.color.g = 255;
			material.color.b = 255;

			return material;
		},
		[&ldos, sigma, windowSize](Mat &canvas, const Index &index){
			vector<double> data;
			vector<double> axis;
			double lowerBound = ldos.getLowerBound();
			double upperBound = ldos.getUpperBound();
			unsigned int resolution = ldos.getResolution();
			double dE = (upperBound - lowerBound)/resolution;
			for(int n = 0; n < (int)ldos.getResolution(); n++){
				data.push_back(ldos(index, n));
				axis.push_back(lowerBound + n*dE);
			}

			Plot::Plotter plotter;
			plotter.setCanvas(canvas);
			if(sigma != 0){
				double scaledSigma = sigma/(ldos.getUpperBound() - ldos.getLowerBound())*ldos.getResolution();
				data = Smooth::gaussian(data, scaledSigma, windowSize);
			}
			plotter.plot(axis, data);

			int baseLine;
			Size size = getTextSize(
				index.toString(),
				FONT_HERSHEY_SIMPLEX,
				0.5,
				1,
				&baseLine
			);
			putText(
				canvas,
				index.toString(),
				Point(
					canvas.cols - size.width - 10,
					size.height + 10
				),
				FONT_HERSHEY_SIMPLEX,
				0.5,
				Scalar(0, 0, 0),
				1,
				true
			);
		}
	);
}

void RayTracer::render(
	const IndexDescriptor &indexDescriptor,
	const Model &model,
	const vector<const FieldWrapper*> &fields,
	function<Material(HitDescriptor &hitDescriptor)> &&lambdaColorPicker,
	function<void(Mat &canvas, const Index &index)> &&lambdaInteractive
){
	//Setup viewport.
	const Vector3d &cameraPosition = renderContext.getCameraPosition();
	const Vector3d &focus = renderContext.getFocus();
	const Vector3d &up = renderContext.getUp();
	unsigned int width = renderContext.getWidth();
	unsigned int height = renderContext.getHeight();
	Vector3d unitY = up.unit();
	Vector3d unitX = ((focus - cameraPosition)*up).unit();
	unitY = (unitX*(focus - cameraPosition)).unit();
	double scaleFactor = (focus - cameraPosition).norm()/(double)width;

	//Get geometry.
	const Geometry &geometry = model.getGeometry();

	//Setup IndexTree.
	const IndexTree &indexTree = indexDescriptor.getIndexTree();
/*	IndexTree::Iterator iterator = indexTree.begin();
	vector<Vector3d> coordinates;
	while(!iterator.getHasReachedEnd()){
		Index index = iterator.getIndex();

		Index i = index;
		for(unsigned int n = 0; n < i.getSize(); n++)
			if(i.at(n) < 0)
				i.at(n) = IDX_ALL;

		vector<Index> indices = model.getHoppingAmplitudeSet().getIndexList(i);

		coordinates.push_back(Vector3d({0., 0., 0.}));
		for(unsigned int n = 0; n < indices.size(); n++){
			const double *c = geometry->getCoordinates(indices.at(n));
			coordinates.back().x += c[0]/indices.size();
			coordinates.back().y += c[1]/indices.size();
			coordinates.back().z += c[2]/indices.size();
		}

		iterator.searchNext();
	}*/
	vector<Vector3d> coordinates;
	for(
		IndexTree::ConstIterator iterator = indexTree.cbegin();
		iterator != indexTree.cend();
		++iterator
	){
		Index index = *iterator;

		Index i = index;
		for(unsigned int n = 0; n < i.getSize(); n++)
			if(i.at(n) < 0)
				i.at(n) = IDX_ALL;

		vector<Index> indices = model.getHoppingAmplitudeSet().getIndexList(i);

		coordinates.push_back(Vector3d({0., 0., 0.}));
		for(unsigned int n = 0; n < indices.size(); n++){
//			const double *c = geometry->getCoordinates(indices.at(n));
			const vector<double> &c = geometry.getCoordinate(indices.at(n));
			coordinates.back().x += c[0]/indices.size();
			coordinates.back().y += c[1]/indices.size();
			coordinates.back().z += c[2]/indices.size();
		}
	}

	//Setup canvas and HitDescriptors.
	if(renderResult != nullptr)
		delete renderResult;
	renderResult = new RenderResult(width, height);
	Mat canvas = renderResult->getCanvas();
	vector<HitDescriptor> **hitDescriptors = renderResult->getHitDescriptors();

	//Render.
	for(unsigned int x = 0; x < width; x++){
		for(unsigned int y = 0; y < height; y++){
			Vector3d target = focus
				+ (scaleFactor*((double)x - width/2.))*unitX
				+ (scaleFactor*((double)y - height/2.))*unitY;
			Vector3d rayDirection = (target - cameraPosition).unit();

			Color color = trace(
				coordinates,
				cameraPosition,
				rayDirection,
				indexTree,
				fields,
				hitDescriptors[x][y],
				lambdaColorPicker,
				renderContext.getNumDeflections()
			);

			canvas.at<Vec3f>(height - 1 - y, x)[0] = color.b;
			canvas.at<Vec3f>(height - 1 - y, x)[1] = color.g;
			canvas.at<Vec3f>(height - 1 - y, x)[2] = color.r;
		}
	}

	//Run interactive mode if lambdaInteractive is defined.
	if(lambdaInteractive){
		namedWindow("Traced image", WINDOW_AUTOSIZE);
		namedWindow("Property window");
		Mat image = getCharImage();
		imshow("Traced image", image);
		Mat propertyCanvas = Mat::zeros(400, 600, CV_8UC3);
		TBTKAssert(
			EventHandler::lock(
				this,
				[&lambdaInteractive, &hitDescriptors, &propertyCanvas, height, width](
					int event,
					int x,
					int y,
					int flags,
					void *userData
				){
					Plot::Plotter plotter;
					plotter.setCanvas(propertyCanvas);
					if(hitDescriptors[x][height - 1 - y].size() > 0){
						const Index& index = hitDescriptors[x][height - 1 -y].at(0).getIndex();
						lambdaInteractive(propertyCanvas, index);
					}
					imshow("Property window", propertyCanvas);
				}
			),
			"RayTracer::render()",
			"Unable to get lock from EventHandler.",
			""
		);
		setMouseCallback(
			"Traced image",
			EventHandler::onMouseChange,
			NULL
		);

		bool done = false;
		while(!done){
			char key = waitKey(0);
			switch(key){
			case 'q':
			case 'Q':
			case 27:	//ESC
				done = true;
				break;
			default:
				Streams::out << key << "\n";
				break;
			}
		}
	}
}

RayTracer::Color RayTracer::trace(
	const vector<Vector3d> &coordinates,
	const Vector3d &raySource,
	const Vector3d &rayDirection,
	const IndexTree &indexTree,
	const vector<const FieldWrapper*> &fields,
	vector<HitDescriptor> &hitDescriptors,
	function<Material(HitDescriptor &hitDescriptor)> lambdaColorPicker,
	unsigned int numDeflections
){
	//Parameters.
	double stateRadius = renderContext.getStateRadius();

	//Trace ray and determine hit objects.
	vector<unsigned int> hits;
	for(unsigned int n = 0; n < coordinates.size(); n++){
		if(
			((coordinates.at(n) - raySource)*rayDirection).norm() < stateRadius
			&& Vector3d::dotProduct(coordinates.at(n) - raySource, rayDirection) > 0
		){
			hits.push_back(n);
		}
	}

	Color color;
	color.r = 0;
	color.g = 0;
	color.b = 0;
	if(hits.size() > 0){
		double minDistance = (coordinates.at(hits.at(0)) - raySource).norm();
		unsigned int minDistanceIndex = hits.at(0);
		for(unsigned int n = 1; n < hits.size(); n++){
			double distance = (coordinates.at(hits.at(n)) - raySource).norm();
			if(distance < minDistance){
				minDistance = distance;
				minDistanceIndex = hits.at(n);
			}
		}

		HitDescriptor hitDescriptor(renderContext);
		hitDescriptor.setRaySource(raySource);
		hitDescriptor.setRayDirection(
			rayDirection
		);
		hitDescriptor.setIndex(
			indexTree.getPhysicalIndex(
				minDistanceIndex
			)
		);
		hitDescriptor.setCoordinate(
			coordinates.at(
				minDistanceIndex
			)
		);
		hitDescriptors.push_back(hitDescriptor);

		if(fields.size() > 0){
			Color fieldColor = traceFields(
				fields,
				raySource,
				hitDescriptor.getImpactPosition()
			);
			color.r += fieldColor.r;
			color.g += fieldColor.g;
			color.b += fieldColor.b;
		}

		Material material = lambdaColorPicker(
			hitDescriptor
		);

		Vector3d directionFromObject = hitDescriptor.getDirectionFromObject();
		double lightProjection = Vector3d::dotProduct(
			directionFromObject.unit(),
			Vector3d({0, 0, 1})
		);
		color.r = material.color.r*(material.ambient + material.diffusive*lightProjection)/(material.ambient + material.diffusive);
		color.g = material.color.g*(material.ambient + material.diffusive*lightProjection)/(material.ambient + material.diffusive);
		color.b = material.color.b*(material.ambient + material.diffusive*lightProjection)/(material.ambient + material.diffusive);

		hitDescriptor.getDirectionFromObject();
		if(numDeflections != 0){
			vector<HitDescriptor> specularHitDescriptors;
			const Vector3d &impactPosition = hitDescriptor.getImpactPosition();
			Vector3d newDirection =	(rayDirection - 2*hitDescriptor.getDirectionFromObject()*Vector3d::dotProduct(
				hitDescriptor.getDirectionFromObject(), rayDirection
			)).unit();
			Color specularColor = trace(
				coordinates,
				impactPosition,
				newDirection,
				indexTree,
				fields,
				specularHitDescriptors,
				lambdaColorPicker,
				numDeflections - 1
			);

			color.r = color.r*(1 - material.specular) + material.specular*specularColor.r;
			color.g = color.g*(1 - material.specular) + material.specular*specularColor.g;
			color.b = color.b*(1 - material.specular) + material.specular*specularColor.b;
		}
	}
	else if(fields.size() > 0){
		Color fieldColor = traceFields(
			fields,
			raySource,
			raySource + rayDirection*renderContext.getRayLength()
		);
		color.r = fieldColor.r;
		color.g = fieldColor.g;
		color.b = fieldColor.b;
	}

	return color;
}

RayTracer::Color RayTracer::traceFields(
	const vector<const FieldWrapper*> &fields,
	const Vector3d &raySource,
	const Vector3d &rayEnd
){
	Color color;
	color.r = 0;
	color.g = 0;
	color.b = 0;

	for(unsigned int n = 0; n < fields.size(); n++){
		const FieldWrapper *field = fields.at(n);
		FieldWrapper::DataType dataType = field->getDataType();
		FieldWrapper::ArgumentType argumentType = field->getArgumentType();
		TBTKAssert(
			argumentType == FieldWrapper::ArgumentType::Double,
			"RayTracer::traceField()",
			"Unable to trace fields with the given argument type.",
			""
		);
		switch(dataType){
		case FieldWrapper::DataType::ComplexDouble:
		{
			Vector3d coordinates(field->getCoordinates<complex<double>, double>());
			double extent = field->getExtent<complex<double>, double>();
			Vector3d rayDirection = (rayEnd - raySource).unit();
			if(((coordinates - raySource)*rayDirection).norm() < extent){
				Vector3d v = raySource;
				Vector3d dv = (rayEnd - raySource)/renderContext.getNumRaySegments();
				double squaredExtent = extent*extent;
				for(unsigned int n = 0; n < renderContext.getNumRaySegments(); n++){
					v = v + dv;
					if(Vector3d::dotProduct(v - coordinates, v - coordinates) > squaredExtent)
						continue;
					complex<double> amplitude = field->operator()<complex<double>, double>({v.x, v.y, v.z});
					double value = amplitude.real()*amplitude.real() + amplitude.imag()*amplitude.imag();
					color.r += value*(255 - color.r);
					color.g += value*(255 - color.g);
					color.b += value*(255 - color.b);
				}
			}
			break;
		}
		default:
			TBTKExit(
				"RayTracer::traceField()",
				"Unable to trace field with the given data type.",
				""
			);
		}
	}

	return color;
}

Mat RayTracer::getCharImage() const{
	TBTKAssert(
		renderResult != nullptr,
		"RayTracer::getCharImage()",
		"Render result missing.",
		"First render an image."
	);

	Mat &canvas = renderResult->getCanvas();

	double minValue = canvas.at<Vec3f>(0, 0)[0];
	double maxValue = canvas.at<Vec3f>(0, 0)[0];
	for(int x = 0; x < canvas.cols; x++){
		for(int y = 0; y < canvas.rows; y++){
			for(int n = 0; n < 3; n++){
				if(canvas.at<Vec3f>(y, x)[n] < minValue)
					minValue = canvas.at<Vec3f>(y, x)[n];
				if(canvas.at<Vec3f>(y, x)[n] > maxValue)
					maxValue = canvas.at<Vec3f>(y, x)[n];
			}
		}
	}

//	vector<HitDescriptor>** hitDescriptors = renderResult->getHitDescriptors();
	Mat image = Mat::zeros(canvas.rows, canvas.cols, CV_8UC3);
	for(int x = 0; x < canvas.cols; x++){
		for(int y = 0; y < canvas.rows; y++){
			for(unsigned int n = 0; n < 3; n++){
//				if(hitDescriptors[x][y].size() > 0)
//					image.at<Vec3b>(canvas.rows - 1 - y, x)[n] = 255*((canvas.at<Vec3f>(canvas.rows - 1 - y, x)[n] - minValue)/(maxValue - minValue));
//				else
//					image.at<Vec3b>(canvas.rows - 1 - y, x)[n] = 0;
					image.at<Vec3b>(canvas.rows - 1 - y, x)[n] = canvas.at<Vec3f>(canvas.rows - 1 - y, x)[n];
			}
		}
	}

	return image;
}

void RayTracer::save(string filename){
	Mat image = getCharImage();
	imwrite(filename, image);
}

RayTracer::RenderContext::RenderContext(){
	cameraPosition = Vector3d({0, 0, 10});
	focus = Vector3d({0, 0, 0});
	up = Vector3d({0, 1, 0});

	width = 600;
	height = 400;

	stateRadius = 0.5;
	numDeflections = 0;
}

RayTracer::RenderContext::~RenderContext(){
}

RayTracer::HitDescriptor::HitDescriptor(const RenderContext &renderContext){
	this->renderContext = &renderContext;
	directionFromObject = nullptr;
	impactPosition = nullptr;
}

RayTracer::HitDescriptor::HitDescriptor(
	const HitDescriptor &hitDescriptor
) :
	renderContext(hitDescriptor.renderContext),
	rayDirection(hitDescriptor.rayDirection),
	index(hitDescriptor.index),
	coordinate(hitDescriptor.coordinate)
{
	if(hitDescriptor.directionFromObject == nullptr){
		directionFromObject = nullptr;
	}
	else{
		directionFromObject = new Vector3d(
			*hitDescriptor.directionFromObject
		);
	}

	if(hitDescriptor.impactPosition == nullptr){
		impactPosition = nullptr;
	}
	else{
		impactPosition = new Vector3d(
			*hitDescriptor.impactPosition
		);
	}
}

RayTracer::HitDescriptor::HitDescriptor(
	HitDescriptor &&hitDescriptor
) :
	renderContext(std::move(hitDescriptor.renderContext)),
	rayDirection(std::move(hitDescriptor.rayDirection)),
	index(std::move(hitDescriptor.index)),
	coordinate(std::move(hitDescriptor.coordinate))
{
	if(hitDescriptor.directionFromObject == nullptr){
		directionFromObject = nullptr;
	}
	else{
		directionFromObject = hitDescriptor.directionFromObject;
		hitDescriptor.directionFromObject = nullptr;
	}

	if(hitDescriptor.impactPosition == nullptr){
		impactPosition = nullptr;
	}
	else{
		impactPosition = hitDescriptor.impactPosition;
		hitDescriptor.impactPosition = nullptr;
	}
}

RayTracer::HitDescriptor::~HitDescriptor(){
	if(directionFromObject != nullptr)
		delete directionFromObject;
	if(impactPosition != nullptr)
		delete impactPosition;
}

RayTracer::HitDescriptor& RayTracer::HitDescriptor::operator=(
	const HitDescriptor &rhs
){
	if(this != &rhs){
		renderContext = rhs.renderContext;
		rayDirection = rhs.rayDirection;
		index = rhs.index;
		coordinate = rhs.coordinate;

		if(rhs.directionFromObject == nullptr)
			directionFromObject = nullptr;
		else
			directionFromObject = new Vector3d(*rhs.directionFromObject);

		if(rhs.impactPosition == nullptr)
			impactPosition = nullptr;
		else
			impactPosition = new Vector3d(*rhs.impactPosition);
	}

	return *this;
}

RayTracer::HitDescriptor& RayTracer::HitDescriptor::operator=(
	HitDescriptor &&rhs
){
	if(this != &rhs){
		renderContext = rhs.renderContext;
		rayDirection = rhs.rayDirection;
		index = rhs.index;
		coordinate = rhs.coordinate;

		if(rhs.directionFromObject == nullptr){
			directionFromObject = nullptr;
		}
		else{
			directionFromObject = rhs.directionFromObject;
			rhs.directionFromObject = nullptr;
		}

		if(rhs.impactPosition == nullptr){
			impactPosition = nullptr;
		}
		else{
			impactPosition = rhs.impactPosition;
			rhs.impactPosition = nullptr;
		}
	}

	return *this;
}

const Vector3d& RayTracer::HitDescriptor::getImpactPosition(){
	if(impactPosition != nullptr)
		return *impactPosition;

	impactPosition = new Vector3d(
		coordinate + renderContext->getStateRadius()*getDirectionFromObject()
	);

	return *impactPosition;
}

const Vector3d& RayTracer::HitDescriptor::getDirectionFromObject(){
	if(directionFromObject != nullptr)
		return *directionFromObject;

	double stateRadius = renderContext->getStateRadius();

	//Here v is the vector from the object to the camera, t is the unit
	//vector in the direction of the ray, and lamvda*t is the vector from
	//the camera to the hit position.
	Vector3d v = coordinate - raySource;
	double a = Vector3d::dotProduct(v, rayDirection);
	double b = Vector3d::dotProduct(v, v);
	double lambda = a - sqrt(stateRadius*stateRadius + a*a - b);
	Vector3d hitPoint = raySource + lambda*rayDirection;

	directionFromObject = new Vector3d((hitPoint - coordinate).unit());

	return *directionFromObject;
}

RayTracer::RenderResult::RenderResult(unsigned int width, unsigned int height){
	this->width = width;
	this->height = height;

	canvas = Mat::zeros(height, width, CV_32FC3);

	hitDescriptors = new vector<HitDescriptor>*[width];
	for(unsigned int x = 0; x < width; x++)
		hitDescriptors[x] = new vector<HitDescriptor>[height];
}

RayTracer::RenderResult::RenderResult(const RenderResult &renderResult){
	width = renderResult.width;
	height = renderResult.height;
	canvas = renderResult.canvas.clone();

	hitDescriptors = new vector<HitDescriptor>*[width];
	for(unsigned int x = 0; x < width; x++){
		hitDescriptors[x] = new vector<HitDescriptor>[height];
		for(unsigned int y = 0; y < height; y++)
			for(unsigned int n = 0; n < renderResult.hitDescriptors[x][y].size(); n++)
				hitDescriptors[x][y].push_back(renderResult.hitDescriptors[x][y].at(n));
	}
}

RayTracer::RenderResult::RenderResult(RenderResult &&renderResult){
	width = renderResult.width;
	height = renderResult.height;
	//It may be possible to improve performance by removing .clone().
	//However, it is not clear to me whether this would result in a memory
	//leak or not. If openCV performs proper reference counting on the
	//pointer type data of Mat, removing .clone() should be fine.
	canvas = renderResult.canvas.clone();

	hitDescriptors = renderResult.hitDescriptors;
	renderResult.hitDescriptors = nullptr;
}

RayTracer::RenderResult::~RenderResult(){
	if(hitDescriptors != nullptr){
		for(unsigned int x = 0; x < width; x++)
			delete [] hitDescriptors[x];
		delete [] hitDescriptors;
	}
}

RayTracer::RenderResult& RayTracer::RenderResult::operator=(
	const RenderResult &rhs
){
	if(this != &rhs){
		width = rhs.width;
		height = rhs.height;
		canvas = rhs.canvas.clone();

		hitDescriptors = new vector<HitDescriptor>*[width];
		for(unsigned int x = 0; x < width; x++){
			hitDescriptors[x] = new vector<HitDescriptor>[height];
			for(unsigned int y = 0; y < height; y++)
				for(unsigned int n = 0; n < rhs.hitDescriptors[x][y].size(); n++)
					hitDescriptors[x][y].push_back(rhs.hitDescriptors[x][y].at(n));
		}
	}

	return *this;
}

RayTracer::RenderResult& RayTracer::RenderResult::operator=(
	RenderResult &&rhs
){
	if(this != &rhs){
		width = rhs.width;
		height = rhs.height;
		//It may be possible to improve performance by removing .clone().
		//However, it is not clear to me whether this would result in a memory
		//leak or not. If openCV performs proper reference counting on the
		//pointer type data of Mat, removing .clone() should be fine.
		canvas = rhs.canvas.clone();

		hitDescriptors = rhs.hitDescriptors;
		rhs.hitDescriptors = nullptr;
	}

	return *this;
}

};	//End of namespace TBTK
