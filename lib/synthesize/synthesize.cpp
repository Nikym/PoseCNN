#include "synthesize.hpp"

using namespace df;

void writeHalfPrecisionVertMap(const std::string filename, const float* vertMap, const int N) {

    std::cout << "writing " << filename << std::endl;

    std::vector<half> halfVertMap(N*3);
    for (int i=0; i<N; ++i) {
        halfVertMap[3*i  ] = half(vertMap[3*i]);
        halfVertMap[3*i+1] = half(vertMap[3*i+1]);
        halfVertMap[3*i+2] = half(vertMap[3*i+2]);
    }

    std::ofstream vertStream;
    vertStream.open(filename,std::ios_base::out | std::ios_base::binary);

    vertStream.write(reinterpret_cast<const char *>(halfVertMap.data()),halfVertMap.size()*sizeof(half));

    vertStream.close();

}

Synthesizer::Synthesizer(std::string model_file, std::string pose_file)
{
  model_file_ = model_file;
  pose_file_ = pose_file;
  counter_ = 0;
  setup_ = 0;
}

void Synthesizer::setup()
{
  int width = 640;
  int height = 480;
  create_window(width, height);

  loadModels(model_file_);
  std::cout << "loaded models" << std::endl;

  loadPoses(pose_file_);
  std::cout << "loaded poses" << std::endl;

  // create tensors

  // labels
  labels_device_ = new df::ManagedDeviceTensor2<int>({width, height});

  // depth map
  depth_map_ = new ManagedTensor<2, float>({width, height});
  depth_map_device_ = new ManagedTensor<2, float, DeviceResident>(depth_map_->dimensions());
  depth_factor_ = 1000.0;
  depth_cutoff_ = 20.0;

  // 3D points
  vertex_map_device_ = new ManagedDeviceTensor2<Vec3>({width, height});
  vertex_map_ = new ManagedHostTensor2<Vec3>({width, height});
  predicted_verts_device_ = new ManagedDeviceTensor2<Eigen::UnalignedVec4<float> > ({width, height});
  predicted_normals_device_ = new ManagedDeviceTensor2<Eigen::UnalignedVec4<float> > ({width, height});
  predicted_verts_ = new ManagedHostTensor2<Eigen::UnalignedVec4<float> >({width, height});
  predicted_normals_ = new ManagedHostTensor2<Eigen::UnalignedVec4<float> >({width, height});

  setup_ = 1;
}

Synthesizer::~Synthesizer()
{
  destroy_window();
}

// create window
void Synthesizer::create_window(int width, int height)
{
  pangolin::CreateWindowAndBind("Synthesizer", width, height);

  gtView_ = &pangolin::Display("gt").SetAspect(640.0/480.);

  // create render
  renderer_ = new df::GLRenderer<df::CanonicalVertRenderType>(width, height);
  renderer_vn_ = new df::GLRenderer<df::VertAndNormalRenderType>(width, height);
}


void Synthesizer::destroy_window()
{
  pangolin::DestroyWindow("Synthesizer");
  delete renderer_;
  delete renderer_vn_;
}

// read the poses
void Synthesizer::loadPoses(const std::string filename)
{
  std::ifstream stream(filename);
  std::vector<std::string> model_names;
  std::string name;

  while ( std::getline (stream, name) )
  {
    std::cout << name << std::endl;
    model_names.push_back(name);
  }
  stream.close();

  // load poses
  const int num_models = model_names.size();
  poses_.resize(num_models);
  pose_nums_.resize(num_models);

  for (int m = 0; m < num_models; ++m)
  {
    // cout lines
    int num_lines = 0;
    std::ifstream stream1(model_names[m]);
    std::string name;

    while ( std::getline (stream1, name) )
      num_lines++;
    stream1.close();
    pose_nums_[m] = num_lines;

    // allocate memory
    float* pose = (float*)malloc(sizeof(float) * num_lines * 7);

    // load data
    FILE* fp = fopen(model_names[m].c_str(), "r");
    for (int i = 0; i < num_lines * 7; i++)
      fscanf(fp, "%f", pose + i);
    fclose(fp);

    poses_[m] = pose;

    std::cout << model_names[m] << std::endl;
  }
}

// read the 3D models
void Synthesizer::loadModels(const std::string filename)
{
  std::ifstream stream(filename);
  std::vector<std::string> model_names;
  std::vector<std::string> texture_names;
  std::string name;

  while ( std::getline (stream, name) )
  {
    std::cout << name << std::endl;
    model_names.push_back(name);
  }
  stream.close();

  // load meshes
  const int num_models = model_names.size();
  assimpMeshes_.resize(num_models);
  texture_names.resize(num_models);

  for (int m = 0; m < num_models; ++m)
  {
    assimpMeshes_[m] = loadTexturedMesh(model_names[m], texture_names[m]);
    std::cout << texture_names[m] << std::endl;
  }

  // buffers
  texturedVertices_.resize(num_models);
  canonicalVertices_.resize(num_models);
  vertexColors_.resize(num_models);
  vertexNormals_.resize(num_models);
  texturedIndices_.resize(num_models);
  texturedCoords_.resize(num_models);
  texturedTextures_.resize(num_models);
  is_textured_.resize(num_models);

  for (int m = 0; m < num_models; m++)
  {
    bool is_textured;
    if (texture_names[m] == "")
      is_textured = false;
    else
      is_textured = true;
    is_textured_[m] = is_textured;

    initializeBuffers(m, assimpMeshes_[m], texture_names[m], texturedVertices_[m], canonicalVertices_[m], vertexColors_[m], vertexNormals_[m],
                      texturedIndices_[m], texturedCoords_[m], texturedTextures_[m], is_textured);
  }
}

aiMesh* Synthesizer::loadTexturedMesh(const std::string filename, std::string & texture_name)
{
    const struct aiScene * scene = aiImportFile(filename.c_str(), aiProcess_JoinIdenticalVertices);
    if (scene == 0) {
        throw std::runtime_error("error: " + std::string(aiGetErrorString()));
    }

    if (scene->mNumMeshes != 1) {
        const int nMeshes = scene->mNumMeshes;
        aiReleaseImport(scene);
        throw std::runtime_error("there are " + std::to_string(nMeshes) + " meshes in " + filename);
    }

    if (!scene->HasMaterials()) {
        throw std::runtime_error(filename + " has no materials");
    }

    std::cout << scene->mNumMaterials << " materials" << std::endl;

    std::string textureName = filename.substr(0,filename.find_last_of('/')+1);
    for (int i = 0; i < scene->mNumMaterials; ++i) 
    {
        aiMaterial * material = scene->mMaterials[i];
        std::cout << "diffuse: " << material->GetTextureCount(aiTextureType_DIFFUSE) << std::endl;
        std::cout << "specular: " << material->GetTextureCount(aiTextureType_SPECULAR) << std::endl;
        std::cout << "ambient: " << material->GetTextureCount(aiTextureType_AMBIENT) << std::endl;
        std::cout << "shininess: " << material->GetTextureCount(aiTextureType_SHININESS) << std::endl;

        if (material->GetTextureCount(aiTextureType_DIFFUSE)) 
        {
            aiString path;
            material->GetTexture(aiTextureType_DIFFUSE,0,&path);
            textureName = textureName + std::string(path.C_Str());
        }
    }

    aiMesh * assimpMesh = scene->mMeshes[0];
    std::cout << "number of vertices: " << assimpMesh->mNumVertices << std::endl;
    std::cout << "number of faces: " << assimpMesh->mNumFaces << std::endl;

    if (!assimpMesh->HasTextureCoords(0))
      texture_name = "";
    else
      texture_name = textureName;

    return assimpMesh;
}


void Synthesizer::initializeBuffers(int model_index, aiMesh* assimpMesh, std::string textureName,
  pangolin::GlBuffer & vertices, pangolin::GlBuffer & canonicalVertices, pangolin::GlBuffer & colors, pangolin::GlBuffer & normals,
  pangolin::GlBuffer & indices, pangolin::GlBuffer & texCoords, pangolin::GlTexture & texture, bool is_textured)
{
    std::cout << "number of vertices: " << assimpMesh->mNumVertices << std::endl;
    std::cout << "number of faces: " << assimpMesh->mNumFaces << std::endl;
    vertices.Reinitialise(pangolin::GlArrayBuffer, assimpMesh->mNumVertices, GL_FLOAT, 3, GL_STATIC_DRAW);
    vertices.Upload(assimpMesh->mVertices, assimpMesh->mNumVertices*sizeof(float)*3);

    // normals
    normals.Reinitialise(pangolin::GlArrayBuffer, assimpMesh->mNumVertices, GL_FLOAT, 3, GL_STATIC_DRAW);
    normals.Upload(assimpMesh->mNormals, assimpMesh->mNumVertices*sizeof(float)*3);

    // canonical vertices
    std::vector<float3> canonicalVerts(assimpMesh->mNumVertices);
    std::memcpy(canonicalVerts.data(), assimpMesh->mVertices, assimpMesh->mNumVertices*sizeof(float3));
    for (std::size_t i = 0; i < assimpMesh->mNumVertices; i++)
        canonicalVerts[i].x += model_index;

    canonicalVertices.Reinitialise(pangolin::GlArrayBuffer, assimpMesh->mNumVertices, GL_FLOAT, 3, GL_STATIC_DRAW);
    canonicalVertices.Upload(canonicalVerts.data(), assimpMesh->mNumVertices*sizeof(float3));

    std::vector<uint3> faces3(assimpMesh->mNumFaces);
    for (std::size_t i = 0; i < assimpMesh->mNumFaces; i++) {
        aiFace & face = assimpMesh->mFaces[i];
        if (face.mNumIndices != 3) {
            throw std::runtime_error("not a triangle mesh");
        }
        faces3[i] = make_uint3(face.mIndices[0],face.mIndices[1],face.mIndices[2]);
    }

    indices.Reinitialise(pangolin::GlElementArrayBuffer,assimpMesh->mNumFaces*3,GL_UNSIGNED_INT,3,GL_STATIC_DRAW);
    indices.Upload(faces3.data(),assimpMesh->mNumFaces*sizeof(int)*3);

    if (is_textured)
    {
      std::cout << "loading texture from " << textureName << std::endl;
      texture.LoadFromFile(textureName);

      std::cout << "loading tex coords..." << std::endl;
      texCoords.Reinitialise(pangolin::GlArrayBuffer,assimpMesh->mNumVertices,GL_FLOAT,2,GL_STATIC_DRAW);

      std::vector<float2> texCoords2(assimpMesh->mNumVertices);
      for (std::size_t i = 0; i < assimpMesh->mNumVertices; ++i) {
          texCoords2[i] = make_float2(assimpMesh->mTextureCoords[0][i].x,1.0 - assimpMesh->mTextureCoords[0][i].y);
      }
      texCoords.Upload(texCoords2.data(),assimpMesh->mNumVertices*sizeof(float)*2);
    }
    else
    {
      // vertex colors
      std::vector<float3> colors3(assimpMesh->mNumVertices);
      for (std::size_t i = 0; i < assimpMesh->mNumVertices; i++) 
      {
          aiColor4D & color = assimpMesh->mColors[0][i];
          colors3[i] = make_float3(color.r, color.g, color.b);
      }
      colors.Reinitialise(pangolin::GlArrayBuffer, assimpMesh->mNumVertices, GL_FLOAT, 3, GL_STATIC_DRAW);
      colors.Upload(colors3.data(), assimpMesh->mNumVertices*sizeof(float)*3);
    }
}


void Synthesizer::render(int width, int height, float fx, float fy, float px, float py, float znear, float zfar, 
              unsigned char* color, float* depth, float* vertmap, float* class_indexes, float *poses_return, float* centers_return,
              float* vertex_targets, float* vertex_weights, float weight)
{
  bool is_textured = true;
  int is_save = 0;

  pangolin::OpenGlMatrixSpec projectionMatrix = pangolin::ProjectionMatrixRDF_TopLeft(width, height, fx, fy, px+0.5, py+0.5, znear, zfar);
  pangolin::OpenGlMatrixSpec projectionMatrix_reverse = pangolin::ProjectionMatrixRDF_TopLeft(width, height, fx, -fy, px+0.5, height-(py+0.5), znear, zfar);

  // show gt pose
  glEnable(GL_DEPTH_TEST);
  glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);

  // sample the number of objects in the scene
  int num = irand(5, 9);
  int num_classes = pose_nums_.size();

  // sample object classes
  std::vector<int> class_ids(num);
  for (int i = 0; i < num; )
  {
    int class_id = irand(0, num_classes);
    int flag = 1;
    for (int j = 0; j < i; j++)
    {
      if(class_id == class_ids[j])
      {
        flag = 0;
        break;
      }
    }
    if (flag)
    {
      class_ids[i++] = class_id;
    }
  }

  if (class_indexes)
  {
    for (int i = 0; i < num; i++)
      class_indexes[i] = class_ids[i];
  }

  // store the poses
  std::vector<Sophus::SE3d> poses(num);
  double threshold = 0.2;

  for (int i = 0; i < num; i++)
  {
    int class_id = class_ids[i];

    while(1)
    {
      // sample a pose
      int seed = irand(0, pose_nums_[class_id]);
      float* pose = poses_[class_id] + seed * 7;

      Eigen::Quaterniond quaternion(pose[0], pose[1], pose[2], pose[3]);
      Sophus::SE3d::Point translation(pose[4], pose[5], pose[6]);
      const Sophus::SE3d T_co(quaternion, translation);

      int flag = 1;
      for (int j = 0; j < i; j++)
      {
        Sophus::SE3d::Point T = poses[j].translation() - translation;
        double d = T.norm();
        if (d < threshold)
        {
          flag = 0;
          break;
        }
      }

      if (flag)
      {
        poses[i] = T_co;
        if (poses_return)
        {
          for (int j = 0; j < 7; j++)
            poses_return[i * 7 + j] = pose[j];
        }
        break;
      }
    }
  }

  // render vertmap
  std::vector<Eigen::Matrix4f> transforms(num);
  std::vector<std::vector<pangolin::GlBuffer *> > attributeBuffers(num);
  std::vector<pangolin::GlBuffer*> modelIndexBuffers(num);

  for (int i = 0; i < num; i++)
  {
    int class_id = class_ids[i];
    transforms[i] = poses[i].matrix().cast<float>();
    attributeBuffers[i].push_back(&texturedVertices_[class_id]);
    attributeBuffers[i].push_back(&canonicalVertices_[class_id]);
    modelIndexBuffers[i] = &texturedIndices_[class_id];
  }

  glClearColor(std::nanf(""), std::nanf(""), std::nanf(""), std::nanf(""));
  renderer_->setProjectionMatrix(projectionMatrix_reverse);
  renderer_->render(attributeBuffers, modelIndexBuffers, transforms);

  glColor3f(1, 1, 1);
  gtView_->ActivateScissorAndClear();
  renderer_->texture(0).RenderToViewportFlipY();

  if (vertmap)
  {
    renderer_->texture(0).Download(vertmap, GL_RGB, GL_FLOAT);
    if (is_save)
    {
      std::string filename = std::to_string(counter_) + ".vertmap";
      writeHalfPrecisionVertMap(filename, vertmap, height*width);
    }

    // compute object 2D centers
    std::vector<float> center_x(num_classes, 0);
    std::vector<float> center_y(num_classes, 0);
    for (int i = 0; i < num; i++)
    {
      int class_id = class_ids[i];
      float tx = poses_return[i * 7 + 4];
      float ty = poses_return[i * 7 + 5];
      float tz = poses_return[i * 7 + 6];
      center_x[class_id] = fx * (tx / tz) + px;
      center_y[class_id] = fy * (ty / tz) + py;
    }

    if (centers_return)
    {
      for (int i = 0; i < num_classes; i++)
      {
        centers_return[2 * i] = center_x[i];
        centers_return[2 * i + 1] = center_y[i];
      }
    }

    // compute center regression targets and weights
    for (int x = 0; x < width; x++)
    {
      for (int y = 0; y < height; y++)
      {
        float vx = vertmap[3 * (y * width + x)];
        if (std::isnan(vx))
          continue;
        int label = std::round(vx);
        // object center
        float cx = center_x[label];
        float cy = center_y[label];

        float rx = cx - x;
        float ry = cy - y;
        float norm = std::sqrt(rx * rx + ry * ry) + 1e-10;

        // assign value
        int offset = (label + 1) * 2 + 2 * (num_classes + 1) * (y * width + x);
        vertex_targets[offset] = rx / norm;
        vertex_weights[offset] = weight;

        offset = (label + 1) * 2 + 1 + 2 * (num_classes + 1) * (y * width + x);
        vertex_targets[offset] = ry / norm;
        vertex_weights[offset] = weight;
      }
    }
  }

  GLfloat lightpos0[] = {drand(-1., 1.), drand(-1., 1.), drand(1., 6.), 0.};
  GLfloat lightpos1[] = {drand(-1., 1.), drand(-1., 1.), drand(1., 6.), 0.};

  // render color image
  glColor3ub(255,255,255);
  gtView_->ActivateScissorAndClear();
  for (int i = 0; i < num; i++)
  {
    int class_id = class_ids[i];

    glMatrixMode(GL_PROJECTION);
    projectionMatrix.Load();
    glMatrixMode(GL_MODELVIEW);

    Eigen::Matrix4f mv = poses[i].cast<float>().matrix();
    pangolin::OpenGlMatrix mvMatrix(mv);
    mvMatrix.Load();

    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glLightfv(GL_LIGHT0, GL_POSITION, lightpos0);
    glEnable(GL_LIGHT1);
    glLightfv(GL_LIGHT1, GL_POSITION, lightpos1);

    glEnable(GL_TEXTURE_2D);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);
    texturedTextures_[class_id].Bind();
    texturedVertices_[class_id].Bind();
    glVertexPointer(3,GL_FLOAT,0,0);
    texturedCoords_[class_id].Bind();
    glTexCoordPointer(2,GL_FLOAT,0,0);
    texturedIndices_[class_id].Bind();
    glDrawElements(GL_TRIANGLES, texturedIndices_[class_id].num_elements, GL_UNSIGNED_INT, 0);
    texturedIndices_[class_id].Unbind();
    texturedTextures_[class_id].Unbind();
    texturedVertices_[class_id].Unbind();
    texturedCoords_[class_id].Unbind();
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    glDisable(GL_TEXTURE_2D);

    glDisable(GL_LIGHT0);
    glDisable(GL_LIGHT1);
    glDisable(GL_LIGHTING);
  }

  // read color image
  if (color)
  {
    glReadPixels(0, 0, width, height, GL_BGRA, GL_UNSIGNED_BYTE, color);
    if (is_save)
    {
      cv::Mat C = cv::Mat(height, width, CV_8UC4, color);
      cv::Mat output;
      cv::flip(C, output, 0);
      std::string filename = std::to_string(counter_) + "_color.png";
      cv::imwrite(filename.c_str(), output);
    }
  }
  
  // read depth image
  if (depth)
  {
    glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT, depth);

    if (is_save)
    {
      // write depth
      cv::Mat D = cv::Mat(height, width, CV_32FC1, depth);
      cv::Mat DD = cv::Mat(height, width, CV_16UC1);
      for (int x = 0; x < width; x++)
      { 
        for (int y = 0; y < height; y++)
        {
          if (D.at<float>(y, x) == 1)
            DD.at<short>(y, x) = 0;
          else
            DD.at<short>(y, x) = short(10000 * 2 * zfar * znear / (zfar + znear - (zfar - znear) * (2 * D.at<float>(y, x) - 1)));
        }
      }

      std::string filename = std::to_string(counter_) + "_depth.png";
      cv::Mat output;
      cv::flip(DD, output, 0);
      cv::imwrite(filename.c_str(), output);
    }
  }

  if (is_save)
  {
    std::string filename = std::to_string(counter_++);
    pangolin::SaveWindowOnRender(filename);
  }
  pangolin::FinishFrame();

  counter_++;
}


jp::jp_trans_t Synthesizer::quat2our(const Sophus::SE3d T_co)
{
  Eigen::Matrix4d mv = T_co.matrix();

  // map data types
  cv::Mat rmat(3, 3, CV_64F);
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++)
      rmat.at<double>(i,j) = mv(i, j);
  }

  cv::Point3d tpt(mv(0, 3), mv(1, 3), mv(2, 3));

  // result may be reconstructed behind the camera
  if(cv::determinant(rmat) < 0)
  {
    tpt = -tpt;
    rmat = -rmat;
  }
	
  return jp::jp_trans_t(rmat, tpt);  
}


// render for one class
void Synthesizer::render_one(int which_class, int width, int height, float fx, float fy, float px, float py, float znear, float zfar, 
              unsigned char* color, float* depth, float* vertmap, float *poses_return, float* centers_return, float* extents)
{
  bool is_textured = false;
  int is_save = 0;

  pangolin::OpenGlMatrixSpec projectionMatrix = pangolin::ProjectionMatrixRDF_TopLeft(width, height, fx, fy, px+0.5, py+0.5, znear, zfar);
  pangolin::OpenGlMatrixSpec projectionMatrix_reverse = pangolin::ProjectionMatrixRDF_TopLeft(width, height, fx, -fy, px+0.5, height-(py+0.5), znear, zfar);

  // show gt pose
  glEnable(GL_DEPTH_TEST);
  glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);

  int num_classes = pose_nums_.size();
  // sample the number of objects in the scene
  int num;
  if (irand(0, 8) == 0)
    num = 2;
  else
    num = 1;

  // sample object classes
  std::vector<int> class_ids(num);
  class_ids[0] = which_class;
  if (num > 1)
  {
    while (1)
    {
      int class_id = irand(0, num_classes);
      if (class_id != which_class)
      {
        class_ids[1] = class_id;
        break;
      }
    }
  }

  // sample the poses
  std::vector<Sophus::SE3d> poses(num);

  // sample the target object
  int class_id = class_ids[0];
  int seed = irand(0, pose_nums_[class_id]);
  float* pose = poses_[class_id] + seed * 7;

  Eigen::Quaterniond quaternion_first(pose[0] + drand(-0.2, 0.2), pose[1]  + drand(-0.2, 0.2), pose[2]  + drand(-0.2, 0.2), pose[3] + drand(-0.2, 0.2));
  Sophus::SE3d::Point translation_first(pose[4] + drand(-0.2, 0.2), pose[5]  + drand(-0.2, 0.2), pose[6] + drand(-0.1, 0.1));
  const Sophus::SE3d T_co_first(quaternion_first, translation_first);

  poses[0] = T_co_first;
  if (poses_return)
  {
    poses_return[0] = quaternion_first.w();
    poses_return[1] = quaternion_first.x();
    poses_return[2] = quaternion_first.y();
    poses_return[3] = quaternion_first.z();
    poses_return[4] = translation_first(0);
    poses_return[5] = translation_first(1);
    poses_return[6] = translation_first(2);
  }

  if (num > 1)
  {
    // sample the second object
    class_id = class_ids[1];
    seed = irand(0, pose_nums_[class_id]);
    pose = poses_[class_id] + seed * 7;
    Eigen::Quaterniond quaternion_second(pose[0] + drand(-0.2, 0.2), pose[1]  + drand(-0.2, 0.2), pose[2]  + drand(-0.2, 0.2), pose[3] + drand(-0.2, 0.2));
    Sophus::SE3d::Point translation_second;
    float extent = (extents[3 * (class_id + 1)] + extents[3 * (class_id + 1) + 1] + extents[3 * (class_id + 1) + 2]) / 3;

    // sample the center location
    int flag = irand(0, 2);
    if (flag == 0)
      flag = -1;
    translation_second(0) = translation_first(0) + flag * extent * drand(0.2, 0.4);

    flag = irand(0, 2);
    if (flag == 0)
      flag = -1;
    translation_second(1) = translation_first(1) + flag * extent * drand(0.2, 0.4);

    translation_second(2) = translation_first(2) - drand(0.1, 0.2);
    const Sophus::SE3d T_co_second(quaternion_second, translation_second);
    poses[1] = T_co_second;
  }

  // render vertmap
  std::vector<Eigen::Matrix4f> transforms(num);
  std::vector<std::vector<pangolin::GlBuffer *> > attributeBuffers(num);
  std::vector<pangolin::GlBuffer*> modelIndexBuffers(num);

  for (int i = 0; i < num; i++)
  {
    int class_id = class_ids[i];
    transforms[i] = poses[i].matrix().cast<float>();
    attributeBuffers[i].push_back(&texturedVertices_[class_id]);
    attributeBuffers[i].push_back(&canonicalVertices_[class_id]);
    modelIndexBuffers[i] = &texturedIndices_[class_id];
  }

  glClearColor(std::nanf(""), std::nanf(""), std::nanf(""), std::nanf(""));
  renderer_->setProjectionMatrix(projectionMatrix_reverse);
  renderer_->render(attributeBuffers, modelIndexBuffers, transforms);

  glColor3f(1, 1, 1);
  gtView_->ActivateScissorAndClear();
  renderer_->texture(0).RenderToViewportFlipY();

  if (vertmap)
  {
    renderer_->texture(0).Download(vertmap, GL_RGB, GL_FLOAT);
    if (is_save)
    {
      std::string filename = std::to_string(counter_) + ".vertmap";
      writeHalfPrecisionVertMap(filename, vertmap, height*width);
    }

    // compute object 2D center
    if (centers_return)
    {
      float tx = poses_return[4];
      float ty = poses_return[5];
      float tz = poses_return[6];
      centers_return[0] = fx * (tx / tz) + px;
      centers_return[1] = fy * (ty / tz) + py;
    }
  }

  // render color image
  glColor3ub(255,255,255);
  gtView_->ActivateScissorAndClear();
  for (int i = 0; i < num; i++)
  {
    int class_id = class_ids[i];

    glMatrixMode(GL_PROJECTION);
    projectionMatrix.Load();
    glMatrixMode(GL_MODELVIEW);

    Eigen::Matrix4f mv = poses[i].cast<float>().matrix();
    pangolin::OpenGlMatrix mvMatrix(mv);
    mvMatrix.Load();

    glEnableClientState(GL_VERTEX_ARRAY);
    texturedVertices_[class_id].Bind();
    glVertexPointer(3,GL_FLOAT,0,0);
    glEnableClientState(GL_COLOR_ARRAY);
    vertexColors_[class_id].Bind();
    glColorPointer(3,GL_FLOAT,0,0);
    texturedIndices_[class_id].Bind();
    glDrawElements(GL_TRIANGLES, texturedIndices_[class_id].num_elements, GL_UNSIGNED_INT, 0);
    texturedIndices_[class_id].Unbind();
    texturedVertices_[class_id].Unbind();
    vertexColors_[class_id].Unbind();
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
  }

  // read color image
  if (color)
  {
    glReadPixels(0, 0, width, height, GL_BGRA, GL_UNSIGNED_BYTE, color);
    if (is_save)
    {
      cv::Mat C = cv::Mat(height, width, CV_8UC4, color);
      cv::Mat output;
      cv::flip(C, output, 0);
      std::string filename = std::to_string(counter_) + "_color.png";
      cv::imwrite(filename.c_str(), output);
    }
  }
  
  // read depth image
  if (depth)
  {
    glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT, depth);

    if (is_save)
    {
      // write depth
      cv::Mat D = cv::Mat(height, width, CV_32FC1, depth);
      cv::Mat DD = cv::Mat(height, width, CV_16UC1);
      for (int x = 0; x < width; x++)
      { 
        for (int y = 0; y < height; y++)
        {
          if (D.at<float>(y, x) == 1)
            DD.at<short>(y, x) = 0;
          else
            DD.at<short>(y, x) = short(10000 * 2 * zfar * znear / (zfar + znear - (zfar - znear) * (2 * D.at<float>(y, x) - 1)));
        }
      }

      std::string filename = std::to_string(counter_) + "_depth.png";
      cv::Mat output;
      cv::flip(DD, output, 0);
      cv::imwrite(filename.c_str(), output);
    }
  }

  if (is_save)
  {
    std::string filename = std::to_string(counter_++);
    pangolin::SaveWindowOnRender(filename);
  }
  pangolin::FinishFrame();

  counter_++;
}


// get label lists
void Synthesizer::getLabels(const int* labelmap, std::vector<std::vector<int>>& labels, std::vector<int>& object_ids, int width, int height, int num_classes, int minArea)
{
  for(int i = 0; i < num_classes; i++)
    labels.push_back( std::vector<int>() );

  // for each pixel
  // #pragma omp parallel for
  for(int x = 0; x < width; x++)
  for(int y = 0; y < height; y++)
  {
    int label = labelmap[y * width + x];
    labels[label].push_back(y * width + x);
  }

  for(int i = 1; i < num_classes; i++)
  {
    if (labels[i].size() > minArea)
    {
      object_ids.push_back(i);
    }
  }
}


// get 3D bounding boxes
void Synthesizer::getBb3Ds(const float* extents, std::vector<std::vector<cv::Point3f>>& bb3Ds, int num_classes)
{
  // for each object
  bb3Ds.clear();
  for (int i = 1; i < num_classes; i++)
  {
    cv::Vec<float, 3> extent;
    extent(0) = extents[i * 3];
    extent(1) = extents[i * 3 + 1];
    extent(2) = extents[i * 3 + 2];

    bb3Ds.push_back(getBB3D(extent));
  }
}


inline cv::Point2f Synthesizer::getMode2D(jp::id_t objID, const cv::Point2f& pt, const float* vertmap, int width, int num_classes)
{
  int channel = 2 * objID;
  int offset = channel + 2 * num_classes * (pt.y * width + pt.x);

  jp::coord2_t mode;
  mode(0) = vertmap[offset];
  mode(1) = vertmap[offset + 1];

  return cv::Point2f(mode(0), mode(1));
}


inline bool Synthesizer::samplePoint2D(jp::id_t objID, std::vector<cv::Point2f>& eyePts, std::vector<cv::Point2f>& objPts, const cv::Point2f& pt2D, const float* vertmap, int width, int num_classes)
{
  cv::Point2f obj = getMode2D(objID, pt2D, vertmap, width, num_classes); // read out object coordinate

  eyePts.push_back(pt2D);
  objPts.push_back(obj);

  return true;
}


/**
 * @brief Creates a list of pose hypothesis (potentially belonging to multiple objects) which still have to be processed (e.g. refined).
 * 
 * The method includes all remaining hypotheses of an object if there is still more than one, or if there is only one remaining but it still needs to be refined.
 * 
 * @param hypMap Map of object ID to a list of hypotheses for that object.
 * @param maxIt Each hypotheses should be at least this often refined.
 * @return std::vector< Ransac3D::TransHyp*, std::allocator< void > > List of hypotheses to be processed further.
*/
std::vector<TransHyp*> Synthesizer::getWorkingQueue(std::map<jp::id_t, std::vector<TransHyp>>& hypMap, int maxIt)
{
  std::vector<TransHyp*> workingQueue;
      
  for(auto it = hypMap.begin(); it != hypMap.end(); it++)
  for(int h = 0; h < it->second.size(); h++)
    if(it->second.size() > 1 || it->second[h].refSteps < maxIt) //exclude a hypothesis if it is the only one remaining for an object and it has been refined enough already
      workingQueue.push_back(&(it->second[h]));

  return workingQueue;
}


inline float Synthesizer::point2line(cv::Point2d x, cv::Point2f n, cv::Point2f p)
{
  float n1 = -n.y;
  float n2 = n.x;
  float p1 = p.x;
  float p2 = p.y;
  float x1 = x.x;
  float x2 = x.y;

  return fabs(n1 * (x1 - p1) + n2 * (x2 - p2)) / sqrt(n1 * n1 + n2 * n2);
}


inline void Synthesizer::countInliers2D(TransHyp& hyp, const float * vertmap, const std::vector<std::vector<int>>& labels, float inlierThreshold, int width, int num_classes, int pixelBatch)
{
  // reset data of last RANSAC iteration
  hyp.inlierPts2D.clear();
  hyp.inliers = 0;

  hyp.effPixels = 0; // num of pixels drawn
  hyp.maxPixels += pixelBatch; // max num of pixels to be drawn	

  int maxPt = labels[hyp.objID].size(); // num of pixels of this class
  float successRate = hyp.maxPixels / (float) maxPt; // probability to accept a pixel

  std::mt19937 generator;
  std::negative_binomial_distribution<int> distribution(1, successRate); // lets you skip a number of pixels until you encounter the next pixel to accept

  for(unsigned ptIdx = 0; ptIdx < maxPt;)
  {
    int index = labels[hyp.objID][ptIdx];
    cv::Point2d pt2D(index % width, index / width);
  
    hyp.effPixels++;
  
    // read out object coordinate
    cv::Point2d obj = getMode2D(hyp.objID, pt2D, vertmap, width, num_classes);

    // inlier check
    if(point2line(hyp.center, obj, pt2D) < inlierThreshold)
    {
      hyp.inlierPts2D.push_back(std::pair<cv::Point2d, cv::Point2d>(obj, pt2D)); // store object coordinate - camera coordinate correspondence
      hyp.inliers++; // keep track of the number of inliers (correspondences might be thinned out for speed later)
    }

    // advance to the next accepted pixel
    if(successRate < 1)
      ptIdx += std::max(1, distribution(generator));
    else
      ptIdx++;
  }
}


inline void Synthesizer::updateHyp2D(TransHyp& hyp, int maxPixels)
{
  if(hyp.inlierPts2D.size() < 4) return;
  filterInliers2D(hyp, maxPixels); // limit the number of correspondences
      
  // data conversion
  cv::Point2d center = hyp.center;
  Hypothesis trans(center);	
	
  // recalculate pose
  trans.calcCenter(hyp.inlierPts2D);
  hyp.center = trans.getCenter();
}


inline void Synthesizer::filterInliers2D(TransHyp& hyp, int maxInliers)
{
  if(hyp.inlierPts2D.size() < maxInliers) return; // maximum number not reached, do nothing
      		
  std::vector<std::pair<cv::Point2d, cv::Point2d>> inlierPts; // filtered list of inlier correspondences
	
  // select random correspondences to keep
  for(unsigned i = 0; i < maxInliers; i++)
  {
    int idx = irand(0, hyp.inlierPts2D.size());
	    
    inlierPts.push_back(hyp.inlierPts2D[idx]);
  }
	
  hyp.inlierPts2D = inlierPts;
}


// Hough voting
void Synthesizer::estimateCenter(const int* labelmap, const float* vertmap, const float* extents, int height, int width, int num_classes, int preemptive_batch,
  float fx, float fy, float px, float py, float* outputs, float* gt_poses, int num_gt)
{
  //set parameters, see documentation of GlobalProperties
  int maxIterations = 10000000;
  float minArea = 400; // a hypothesis covering less projected area (2D bounding box) can be discarded (too small to estimate anything reasonable)
  float inlierThreshold3D = 0.5;
  int ransacIterations = 256;  // 256
  int preemptiveBatch = preemptive_batch;  // 1000
  int maxPixels = 1000;  // 1000
  int refIt = 8;  // 8

  // labels
  std::vector<std::vector<int>> labels;
  std::vector<int> object_ids;
  getLabels(labelmap, labels, object_ids, width, height, num_classes, minArea);
  std::cout << "read labels" << std::endl;

  // bb3Ds
  getBb3Ds(extents, bb3Ds_, num_classes);

  if (object_ids.size() == 0)
    return;
		
  // hold for each object a list of pose hypothesis, these are optimized until only one remains per object
  std::map<jp::id_t, std::vector<TransHyp>> hypMap;
	
  // sample initial pose hypotheses
  #pragma omp parallel for
  for(unsigned h = 0; h < ransacIterations; h++)
  for(unsigned i = 0; i < maxIterations; i++)
  {
    // camera coordinate - object coordinate correspondences
    std::vector<cv::Point2f> eyePts;
    std::vector<cv::Point2f> objPts;
	    
    // sample first point and choose object ID
    jp::id_t objID = object_ids[irand(0, object_ids.size())];

    if(objID == 0) continue;

    int pindex = irand(0, labels[objID].size());
    int index = labels[objID][pindex];
    cv::Point2f pt1(index % width, index / width);
    
    // sample first correspondence
    if(!samplePoint2D(objID, eyePts, objPts, pt1, vertmap, width, num_classes))
      continue;

    // sample other points in search radius, discard hypothesis if minimum distance constrains are violated
    pindex = irand(0, labels[objID].size());
    index = labels[objID][pindex];
    cv::Point2f pt2(index % width, index / width);

    if(!samplePoint2D(objID, eyePts, objPts, pt2, vertmap, width, num_classes))
      continue;

    // reconstruct camera
    std::vector<std::pair<cv::Point2d, cv::Point2d>> pts2D;
    for(unsigned j = 0; j < eyePts.size(); j++)
    {
      pts2D.push_back(std::pair<cv::Point2d, cv::Point2d>(
      cv::Point2d(objPts[j].x, objPts[j].y),
      cv::Point2d(eyePts[j].x, eyePts[j].y)
      ));
    }

    Hypothesis trans(pts2D);

    // center
    cv::Point2d center = trans.getCenter();
    
    // create a hypothesis object to store meta data
    TransHyp hyp(objID, center);
    
    #pragma omp critical
    {
      hypMap[objID].push_back(hyp);
    }

    break;
  }

  // create a list of all objects where hypptheses have been found
  std::vector<jp::id_t> objList;
  for(std::pair<jp::id_t, std::vector<TransHyp>> hypPair : hypMap)
  {
    objList.push_back(hypPair.first);
  }

  // create a working queue of all hypotheses to process
  std::vector<TransHyp*> workingQueue = getWorkingQueue(hypMap, refIt);
	
  // main preemptive RANSAC loop, it will stop if there is max one hypothesis per object remaining which has been refined a minimal number of times
  while(!workingQueue.empty())
  {
    // draw a batch of pixels and check for inliers, the number of pixels looked at is increased in each iteration
    #pragma omp parallel for
    for(int h = 0; h < workingQueue.size(); h++)
      countInliers2D(*(workingQueue[h]), vertmap, labels, inlierThreshold3D, width, num_classes, preemptiveBatch);
	    	    
    // sort hypothesis according to inlier count and discard bad half
    #pragma omp parallel for 
    for(unsigned o = 0; o < objList.size(); o++)
    {
      jp::id_t objID = objList[o];
      if(hypMap[objID].size() > 1)
      {
	std::sort(hypMap[objID].begin(), hypMap[objID].end());
	hypMap[objID].erase(hypMap[objID].begin() + hypMap[objID].size() / 2, hypMap[objID].end());
      }
    }
    workingQueue = getWorkingQueue(hypMap, refIt);
	    
    // refine
    #pragma omp parallel for
    for(int h = 0; h < workingQueue.size(); h++)
    {
      updateHyp2D(*(workingQueue[h]), maxPixels);
      workingQueue[h]->refSteps++;
    }
    
    workingQueue = getWorkingQueue(hypMap, refIt);
  }

  rois_.clear();
  int count = 0;
  for(auto it = hypMap.begin(); it != hypMap.end(); it++)
  {
    for(int h = 0; h < it->second.size(); h++)
    {
      std::cout << "Estimated Hypothesis for Object " << (int) it->second[h].objID << ":" << std::endl;

      jp::id_t objID = it->second[h].objID;
      cv::Point2d center = it->second[h].center;
      it->second[h].compute_width_height();
      outputs[4 * objID] = center.x;
      outputs[4 * objID + 1] = center.y;
      outputs[4 * objID + 2] = it->second[h].width_;
      outputs[4 * objID + 3] = it->second[h].height_;
    
      std::cout << "Inliers: " << it->second[h].inliers;
      std::printf(" (Rate: %.1f\%)\n", it->second[h].getInlierRate() * 100);
      std::cout << "Center " << center << std::endl;
      std::cout << "---------------------------------------------------" << std::endl;

      // roi (objID, x1, y1, x2, y2, rotation_x, rotation_y, rotation_z, translation_x, translation_y, translation_z, score)
      // generate different pose hypotheses
      std::vector<cv::Vec<float, 12> > poses;
      rois_.push_back(poses);
      int interval = 8;
      for (int rx = 0; rx < interval; rx++)
      {
        for (int ry = 0; ry < interval; ry++)
        {
          for (int rz = 0; rz < interval; rz++)
          {
            cv::Vec<float, 12> roi;
            roi(0) = objID;
            // bounding box
            roi(1) = std::max(center.x - it->second[h].width_ / 2, 0.0);
            roi(2) = std::max(center.y - it->second[h].height_ / 2, 0.0);
            roi(3) = std::min(center.x + it->second[h].width_ / 2, double(width));
            roi(4) = std::min(center.y + it->second[h].height_ / 2, double(height));
            // 6D pose
            roi(5) = 2 * PI * rx / float(interval);
            roi(6) = 2 * PI * ry / float(interval);
            roi(7) = 2 * PI * rz / float(interval);
            // backproject the center
            roi(8) = (center.x - px) / fx;
            roi(9) = (center.y - py) / fy;
            roi(10) = 1.0;
            // score
            roi(11) = -1.0;
            rois_[count].push_back(roi);
          }
        }
      }
      count++;
    }
  }
}


void Synthesizer::visualizePose(int height, int width, float fx, float fy, float px, float py, float znear, float zfar, float* rois, float* outputs, int num_roi)
{
  if (setup_ == 0)
    setup();

  int num = rois_.size();
  pangolin::OpenGlMatrixSpec projectionMatrix = pangolin::ProjectionMatrixRDF_TopLeft(width, height, fx, fy, px+0.5, py+0.5, znear, zfar);

  // render color image
  glEnable(GL_DEPTH_TEST);
  glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
  glColor3ub(255,255,255);
  gtView_->ActivateScissorAndClear();
  float threshold = 0.1;

  for (int i = 0; i < num_roi; i++)
  {
    if (outputs[i * 8 + 7] < threshold)
      continue;
    Eigen::Quaternionf quaternion(outputs[i * 8 + 0], outputs[i * 8 + 1], outputs[i * 8 + 2], outputs[i * 8 + 3]);
    Sophus::SE3f::Point translation(outputs[i * 8 + 4], outputs[i * 8 + 5], outputs[i * 8 + 6]);
    const Sophus::SE3f T_co(quaternion, translation);

    int class_id = int(rois[i * 6 + 1]) - 1;

    glMatrixMode(GL_PROJECTION);
    projectionMatrix.Load();
    glMatrixMode(GL_MODELVIEW);

    Eigen::Matrix4f mv = T_co.cast<float>().matrix();
    pangolin::OpenGlMatrix mvMatrix(mv);
    mvMatrix.Load();

    if (is_textured_[class_id])
    {
      glEnable(GL_TEXTURE_2D);
      glEnableClientState(GL_VERTEX_ARRAY);
      glEnableClientState(GL_TEXTURE_COORD_ARRAY);
      texturedTextures_[class_id].Bind();
      texturedVertices_[class_id].Bind();
      glVertexPointer(3,GL_FLOAT,0,0);
      texturedCoords_[class_id].Bind();
      glTexCoordPointer(2,GL_FLOAT,0,0);
      texturedIndices_[class_id].Bind();
      glDrawElements(GL_TRIANGLES, texturedIndices_[class_id].num_elements, GL_UNSIGNED_INT, 0);
      texturedIndices_[class_id].Unbind();
      texturedTextures_[class_id].Unbind();
      texturedVertices_[class_id].Unbind();
      texturedCoords_[class_id].Unbind();
      glDisableClientState(GL_VERTEX_ARRAY);
      glDisableClientState(GL_TEXTURE_COORD_ARRAY);
      glDisable(GL_TEXTURE_2D);
    }
    else
    {
      glEnableClientState(GL_VERTEX_ARRAY);
      texturedVertices_[class_id].Bind();
      glVertexPointer(3,GL_FLOAT,0,0);
      glEnableClientState(GL_COLOR_ARRAY);
      vertexColors_[class_id].Bind();
      glColorPointer(3,GL_FLOAT,0,0);
      texturedIndices_[class_id].Bind();
      glDrawElements(GL_TRIANGLES, texturedIndices_[class_id].num_elements, GL_UNSIGNED_INT, 0);
      texturedIndices_[class_id].Unbind();
      texturedVertices_[class_id].Unbind();
      vertexColors_[class_id].Unbind();
      glDisableClientState(GL_VERTEX_ARRAY);
      glDisableClientState(GL_COLOR_ARRAY);
    }
  }

  pangolin::FinishFrame();
  std::string filename = std::to_string(counter_++);
  pangolin::SaveWindowOnRender(filename);
}


// ICP
void Synthesizer::solveICP(const int* labelmap, unsigned char* depth, int height, int width, float fx, float fy, float px, float py, float znear, float zfar, 
                           float factor, int num_roi, float* rois, float* poses, float* outputs)
{
  if (setup_ == 0)
    setup();

  // build the camera paramters
  Eigen::Matrix<float,7,1,Eigen::DontAlign> params;
  params[0] = fx;
  params[1] = fy;
  params[2] = px;
  params[3] = py;
  params[4] = 0;
  params[5] = 0;
  params[6] = 0;
  df::Poly3CameraModel<float> model(params);

  // set the depth factor
  depth_factor_ = factor;

  pangolin::OpenGlMatrixSpec projectionMatrix = pangolin::ProjectionMatrixRDF_TopLeft(width, height, fx, -fy, px+0.5, height-(py+0.5), znear, zfar);
  renderer_vn_->setProjectionMatrix(projectionMatrix);

  // for each object
  for(int i = 0; i < num_roi; i++)
  {
    int objID = int(rois[i * 6 + 1]);
    if (objID < 0)
      continue;

    // convert depth values
    float* p = depth_map_->data();
    ushort* q = reinterpret_cast<ushort *>(depth);
    int count = 0;
    for (int j = 0; j < width * height; j++)
    {
      if (labelmap[j] == objID)
      {
        p[j] = q[j] / depth_factor_;
        count++;
      }
      else
        p[j] = 0;
    }

    // backprojection
    depth_map_device_->copyFrom(*depth_map_);
    backproject<float, Poly3CameraModel>(*depth_map_device_, *vertex_map_device_, model);
    vertex_map_->copyFrom(*vertex_map_device_);

    // rendering
    float* pose = poses + i * 7;
    Eigen::Quaternionf quaternion(pose[0], pose[1], pose[2], pose[3]);
    Sophus::SE3f::Point translation(pose[4], pose[5], pose[6]);
    const Sophus::SE3f T_co(quaternion, translation);

    std::vector<pangolin::GlBuffer *> attributeBuffers({&texturedVertices_[objID - 1], &vertexNormals_[objID - 1]});
    renderer_vn_->setModelViewMatrix(T_co.matrix().cast<float>());
    renderer_vn_->render(attributeBuffers, texturedIndices_[objID - 1], GL_TRIANGLES);

    const pangolin::GlTextureCudaArray & vertTex = renderer_vn_->texture(0);
    const pangolin::GlTextureCudaArray & normTex = renderer_vn_->texture(1);

    // copy predicted vertices
    {
      pangolin::CudaScopedMappedArray scopedArray(vertTex);
      cudaMemcpy2DFromArray(predicted_verts_device_->data(), vertTex.width*4*sizeof(float), *scopedArray, 0, 0, vertTex.width*4*sizeof(float), vertTex.height, cudaMemcpyDeviceToDevice);
      predicted_verts_->copyFrom(*predicted_verts_device_);
    }

    // copy predicted normals
    {
      pangolin::CudaScopedMappedArray scopedArray(normTex);
      cudaMemcpy2DFromArray(predicted_normals_device_->data(), normTex.width*4*sizeof(float), *scopedArray, 0, 0, normTex.width*4*sizeof(float), normTex.height, cudaMemcpyDeviceToDevice);
    }

    // Tanner's ICP
    Sophus::SE3f T_co_new = T_co;
    if (count > 400)
    {
      Eigen::Vector2f depthRange(znear, zfar);
      float maxError = 0.02;
      uint icpIterations = 8;

      Sophus::SE3f update = icp(*vertex_map_device_, *predicted_verts_device_, *predicted_normals_device_,
                              model, T_co, depthRange, maxError, icpIterations);

      T_co_new = update * T_co;
      std::cout << T_co_new.matrix3x4() << std::endl;
    }
    std::cout << "class id: " << objID << ", pixels: " << count << std::endl;

    // set output
    Eigen::Quaternionf quaternion_new = T_co_new.unit_quaternion();
    Sophus::SE3f::Point translation_new = T_co_new.translation();

    outputs[i * 7 + 0] = quaternion_new.w();
    outputs[i * 7 + 1] = quaternion_new.x();
    outputs[i * 7 + 2] = quaternion_new.y();
    outputs[i * 7 + 3] = quaternion_new.z();
    outputs[i * 7 + 4] = translation_new(0);
    outputs[i * 7 + 5] = translation_new(1);
    outputs[i * 7 + 6] = translation_new(2);

    glColor3f(1, 1, 1);
    gtView_->ActivateScissorAndClear();
    renderer_vn_->texture(0).RenderToViewportFlipY();
    pangolin::FinishFrame();
  }
}


int main(int argc, char** argv) 
{
  Synthesizer Synthesizer(argv[1], argv[2]);
  Synthesizer.setup();

  // camera parameters
  int width = 640;
  int height = 480;
  float fx = 1066.778, fy = 1067.487, px = 312.9869, py = 241.3109, zfar = 6.0, znear = 0.25;

  while (!pangolin::ShouldQuit()) 
  {
    clock_t start = clock();    

    Synthesizer.render(width, height, fx, fy, px, py, znear, zfar, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 1.0);

    clock_t stop = clock();    
    double elapsed = (double)(stop - start) * 1000.0 / CLOCKS_PER_SEC;
    printf("Time elapsed in ms: %f\n", elapsed);
  }
}