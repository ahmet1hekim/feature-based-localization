#include <SFML/Graphics.hpp>
#include <boost/asio.hpp>
#include <cmath>
#include <iostream>
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <thread>

cv::Mat sfml2opencv(const sf::Image &img) {
  cv::Size size(img.getSize().x, img.getSize().y);
  cv::Mat mat(size, CV_8UC4, (void *)img.getPixelsPtr(), cv::Mat::AUTO_STEP);
  cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB);
  return mat.clone();
}

void sendMat(boost::asio::ip::tcp::socket &socket, const cv::Mat &mat) {
  int32_t rows = mat.rows;
  int32_t cols = mat.cols;
  int32_t type = mat.type();
  boost::asio::write(socket, boost::asio::buffer(&rows, sizeof(rows)));
  boost::asio::write(socket, boost::asio::buffer(&cols, sizeof(cols)));
  boost::asio::write(socket, boost::asio::buffer(&type, sizeof(type)));
  size_t dataSize = mat.total() * mat.elemSize();
  boost::asio::write(socket, boost::asio::buffer(mat.data, dataSize));
}

// ── Autopilot command receiver (port 12347 ← path_planner.py) ────────────────
// N_PATH must match path_planner.py
static constexpr int N_PATH = 50;

// Packet: speed(f), turn(f), est_x(f), est_y(f), goal_x(f), goal_y(f),
// path[N_PATH * 2 floats]
struct AutopilotCmd {
  float speed = 0.f;
  float turn_angle = 0.f;
  float est_x = -1.f;
  float est_y = -1.f;
  float goal_x = 300.f; // defaults match path_planner.py initial values
  float goal_y = 700.f;
  float path_x[N_PATH] = {};
  float path_y[N_PATH] = {};
  int path_len = 0;
};

static AutopilotCmd g_cmd;
static std::mutex g_cmd_mutex;
static bool g_autopilot_active = false;

static void autopilotThread() {
  boost::asio::io_context io;
  boost::asio::ip::tcp::socket sock(io);
  boost::asio::ip::tcp::endpoint ep(boost::asio::ip::make_address("127.0.0.1"),
                                    12347);

  while (true) {
    try {
      std::cout << "[autopilot] Connecting to path planner on port 12347...\n";
      sock.connect(ep);
      std::cout << "[autopilot] Connected to path planner.\n";
      g_autopilot_active = true;

      while (true) {
        // 6 header floats + N_PATH*2 path floats
        constexpr int NFLOATS = 6 + N_PATH * 2;
        float buf[NFLOATS];
        boost::asio::read(sock, boost::asio::buffer(buf, sizeof(buf)));
        std::lock_guard<std::mutex> lk(g_cmd_mutex);
        g_cmd.speed = buf[0];
        g_cmd.turn_angle = buf[1];
        g_cmd.est_x = buf[2];
        g_cmd.est_y = buf[3];
        g_cmd.goal_x = buf[4];
        g_cmd.goal_y = buf[5];
        g_cmd.path_len = 0;
        for (int i = 0; i < N_PATH; ++i) {
          g_cmd.path_x[i] = buf[6 + i * 2];
          g_cmd.path_y[i] = buf[6 + i * 2 + 1];
          if (i > 0 && g_cmd.path_x[i] == g_cmd.path_x[i - 1] &&
              g_cmd.path_y[i] == g_cmd.path_y[i - 1])
            break;
          g_cmd.path_len = i + 1;
        }
      }
    } catch (const std::exception &e) {
      std::cout << "[autopilot] Lost connection: " << e.what()
                << " — retrying in 2s\n";
      g_autopilot_active = false;
      sock.close();
      sock = boost::asio::ip::tcp::socket(io);
      std::this_thread::sleep_for(std::chrono::seconds(2));
    }
  }
}

// ── Overlay geometry helpers
// ────────────────────────────────────────────────── (goal position is now
// received from path_planner.py — no hardcoded constant)

sf::CircleShape makeCircle(float r, sf::Color fill,
                           sf::Color outline = sf::Color::Transparent) {
  sf::CircleShape c(r);
  c.setOrigin({r, r});
  c.setFillColor(fill);
  c.setOutlineColor(outline);
  c.setOutlineThickness(outline == sf::Color::Transparent ? 0.f : 2.f);
  return c;
}

// Draw a dashed line between two world points
void drawDashedLine(sf::RenderTarget &rt, sf::Vector2f a, sf::Vector2f b,
                    sf::Color color, float dashLen = 12.f, float gapLen = 8.f) {
  sf::Vector2f dir = b - a;
  float total = std::sqrt(dir.x * dir.x + dir.y * dir.y);
  if (total < 1.f)
    return;
  sf::Vector2f unit = dir / total;
  float pos = 0.f;
  bool drawing = true;
  while (pos < total) {
    float segLen = drawing ? dashLen : gapLen;
    if (drawing) {
      float end = std::min(pos + segLen, total);
      sf::Vertex line[2] = {{a + unit * pos, color}, {a + unit * end, color}};
      rt.draw(line, 2, sf::PrimitiveType::Lines);
    }
    pos += segLen;
    drawing = !drawing;
  }
}

int main() {
  // ── Port 12345: send frames to SLAM ───────────────────────────────────────
  boost::asio::io_context io_context;
  boost::asio::ip::tcp::acceptor acceptor(
      io_context,
      boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), 12345));

  std::cout << "Waiting for SLAM client on port 12345...\n";
  boost::asio::ip::tcp::socket socket(io_context);
  acceptor.accept(socket);
  std::cout << "SLAM client connected.\n";

  // ── Port 12347: autopilot commands (background thread) ────────────────────
  std::thread autopilot_t(autopilotThread);
  autopilot_t.detach();

  // ── Resolution & assets ───────────────────────────────────────────────────
  const u_int16_t drone_cam_res_x = 960;
  const u_int16_t drone_cam_res_y = 540;
  const u_int16_t window_res_x = 1280;
  const u_int16_t window_res_y = 720;
  const u_int16_t view_res_x = 1920;
  const u_int16_t view_res_y = 1080;

  sf::FileInputStream bg_stream(std::string(ASSETS_DIR) + "/dag.jpg");
  sf::Texture bg_texture(bg_stream);
  sf::Sprite bg_sprite(bg_texture);

  // ── Drone shape ───────────────────────────────────────────────────────────
  float rect_h = 30.f, rect_w = 20.f;
  sf::RectangleShape shape({rect_w, rect_h});
  shape.setFillColor(sf::Color::Red);
  shape.setOrigin({rect_w / 2, rect_h / 2});
  shape.setPosition({static_cast<float>(window_res_x) / 2,
                     static_cast<float>(window_res_y) / 2});

  // ── Overlay shapes ────────────────────────────────────────────────────────
  // Goal marker — position updated each frame from cmd.goal_x/y
  sf::CircleShape goalMarker =
      makeCircle(14.f, sf::Color(255, 220, 0, 200), sf::Color::White);

  // SLAM estimated position: semi-transparent cyan circle
  sf::CircleShape slamMarker =
      makeCircle(10.f, sf::Color(0, 200, 255, 180), sf::Color::White);

  // ── Windows ───────────────────────────────────────────────────────────────
  sf::RenderWindow window(sf::VideoMode({window_res_x, window_res_y}),
                          "Drone Sim", sf::Style::None, sf::State::Windowed);
  window.setVerticalSyncEnabled(true);

  sf::RenderWindow window2(sf::VideoMode({drone_cam_res_x, drone_cam_res_y}),
                           "Drone Cam", sf::Style::None, sf::State::Windowed);
  window2.setVerticalSyncEnabled(true);

  float tomove_y = 0.f;
  float speed_y = 3.5f;

  while (window.isOpen()) {
    // ── Events ────────────────────────────────────────────────────────────
    while (const std::optional event = window.pollEvent()) {
      if (event->is<sf::Event::Closed>())
        window.close();
    }

    // ── Manual keyboard ───────────────────────────────────────────────────
    bool manual_input = false;
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Left)) {
      shape.rotate(sf::degrees(-0.2f));
      manual_input = true;
    }
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Right)) {
      shape.rotate(sf::degrees(0.2f));
      manual_input = true;
    }
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Down)) {
      tomove_y = speed_y;
      manual_input = true;
    }
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Up)) {
      tomove_y = -speed_y;
      manual_input = true;
    }
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Space)) {
      speed_y += 2.5f;
    }

    // ── Autopilot (overridden by keyboard) ────────────────────────────────
    AutopilotCmd cmd;
    {
      std::lock_guard<std::mutex> lk(g_cmd_mutex);
      cmd = g_cmd;
    }
    if (!manual_input && g_autopilot_active) {
      shape.rotate(sf::degrees(cmd.turn_angle));
      tomove_y = cmd.speed;
    }

    // ── Movement + boundary clamping ──────────────────────────────────────
    float rad = shape.getRotation().asDegrees() * 3.14159265f / 180.f;
    sf::Vector2f moveVec(std::sin(rad) * tomove_y, -std::cos(rad) * tomove_y);
    sf::Vector2f futurePos = shape.getPosition() + moveVec;

    sf::Vector2u texSize2 = bg_texture.getSize();
    sf::Vector2f halfSize = shape.getSize() / 2.f;
    if (futurePos.x >= halfSize.x && futurePos.x <= texSize2.x - halfSize.x &&
        futurePos.y >= halfSize.y && futurePos.y <= texSize2.y - halfSize.y)
      shape.move(moveVec);

    tomove_y = 0.f;

    std::cout << shape.getPosition().x << " " << shape.getPosition().y << "\n";

    // ── View: clamp camera so it never shows outside the texture ──────────
    sf::View view1;
    sf::Vector2u texSize = bg_texture.getSize();
    float halfViewW = view_res_x / 2.f, halfViewH = view_res_y / 2.f;
    sf::Vector2f cam = shape.getPosition();
    cam.x = std::max(halfViewW, std::min(cam.x, (float)texSize.x - halfViewW));
    cam.y = std::max(halfViewH, std::min(cam.y, (float)texSize.y - halfViewH));
    view1.setCenter(cam);
    view1.setSize(
        {static_cast<float>(view_res_x), static_cast<float>(view_res_y)});
    window.setView(view1);

    // ── Render map + drone ────────────────────────────────────────────────
    window.draw(bg_sprite);
    window.draw(shape);

    // ── Overlays ─────────────────────────────────────────────────────────

    // 1. Predicted arc (lime-green line + dots)
    if (cmd.path_len > 1) {
      sf::Color arcColor(50, 220, 80, 200);
      for (int i = 1; i < cmd.path_len; ++i) {
        sf::Vertex seg[2] = {{{cmd.path_x[i - 1], cmd.path_y[i - 1]}, arcColor},
                             {{cmd.path_x[i], cmd.path_y[i]}, arcColor}};
        window.draw(seg, 2, sf::PrimitiveType::Lines);
      }
      // Small dot every 5th point
      for (int i = 0; i < cmd.path_len; i += 5) {
        sf::CircleShape dot(2.5f);
        dot.setOrigin({2.5f, 2.5f});
        dot.setFillColor(arcColor);
        dot.setPosition({cmd.path_x[i], cmd.path_y[i]});
        window.draw(dot);
      }
    }

    // 2. Dashed yellow line: drone → goal
    drawDashedLine(window, shape.getPosition(), {cmd.goal_x, cmd.goal_y},
                   sf::Color(255, 220, 0, 160));

    // 3. Goal marker (position driven by Python)
    goalMarker.setPosition({cmd.goal_x, cmd.goal_y});
    window.draw(goalMarker);

    // 3. SLAM estimated position (only if valid)
    if (cmd.est_x > 0.f || cmd.est_y > 0.f) {
      slamMarker.setPosition({cmd.est_x, cmd.est_y});
      window.draw(slamMarker);

      // Dashed line: SLAM estimate → goal (dimmer)
      drawDashedLine(window, {cmd.est_x, cmd.est_y}, {cmd.goal_x, cmd.goal_y},
                     sf::Color(0, 200, 255, 80), 8.f, 6.f);
    }

    window.display();

    // ── Drone-cam: send frame to SLAM ─────────────────────────────────────
    sf::View camView;
    camView.setCenter(shape.getPosition());
    camView.setSize({static_cast<float>(drone_cam_res_x),
                     static_cast<float>(drone_cam_res_y)});
    camView.setRotation(shape.getRotation());
    window2.setView(camView);
    window2.clear();
    window2.draw(bg_sprite);

    sf::Texture winTex({window2.getSize().x, window2.getSize().y});
    winTex.update(window2);
    sf::Image cropped = winTex.copyToImage();
    sendMat(socket, sfml2opencv(cropped));
    float angle = shape.getRotation().asDegrees();
    boost::asio::write(socket, boost::asio::buffer(&angle, sizeof(angle)));
  }
}
