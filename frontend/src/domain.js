export const LINES = ['heart_line', 'head_line', 'life_line'];
export const TOOLS = [
  { id: 'heart_line', name: 'Tâm đạo', color: '#d54c75', hint: 'Nếp phía trên, dưới gốc các ngón. Vẽ từ phía ngón út về ngón cái.' },
  { id: 'head_line', name: 'Trí đạo', color: '#15869c', hint: 'Nếp chạy ngang giữa lòng bàn tay. Vẽ từ phía ngón út về ngón cái.' },
  { id: 'life_line', name: 'Sinh đạo', color: '#a57612', hint: 'Nếp ôm quanh gốc ngón cái. Vẽ từ gần ngón trỏ về cổ tay.' },
  { id: 'width', name: 'Bề rộng', color: '#6650ab', hint: 'Bấm hai mép lòng bàn tay ở cùng mức ngang dưới các ngón; không tính ngón cái.' },
];

export function pointsFor(annotation, tool) {
  return tool === 'width' ? annotation.palm_width_points : annotation.lines[tool].points;
}

export function withPoints(annotation, tool, points) {
  const next = structuredClone(annotation);
  if (tool === 'width') next.palm_width_points = points;
  else next.lines[tool] = { status: points.length === 6 ? 'present' : 'unreviewed', points };
  return next;
}

export function firstIncomplete(annotation) {
  return LINES.find(name => annotation.lines[name].status === 'unreviewed') ||
    (annotation.palm_width_points.length !== 2 ? 'width' : LINES[0]);
}

export function nextPending(images, current) {
  for (let step = 1; step < images.length; step++) {
    const index = (current + step) % images.length;
    if (images[index].status === 'pending') return index;
  }
  return -1;
}

export function swipeSide(distance, threshold = 96) {
  return distance <= -threshold ? 'left' : distance >= threshold ? 'right' : null;
}

// Sample equal distances along the stroke in IMAGE pixels, including both endpoints.
export function sampleStroke(path, width, height, count = 6) {
  if (path.length < 2) return null;
  const cumulative = [0];
  for (let i = 1; i < path.length; i++) {
    cumulative.push(cumulative[i - 1] + Math.hypot(
      (path[i][0] - path[i - 1][0]) * (width - 1),
      (path[i][1] - path[i - 1][1]) * (height - 1),
    ));
  }
  const length = cumulative.at(-1);
  if (length < 8) return null;
  let segment = 1;
  return Array.from({ length: count }, (_, i) => {
    const distance = length * i / (count - 1);
    while (segment < path.length - 1 && cumulative[segment] < distance) segment++;
    const span = cumulative[segment] - cumulative[segment - 1];
    const t = span ? (distance - cumulative[segment - 1]) / span : 0;
    return path[segment - 1].map((value, axis) => value + (path[segment][axis] - value) * t);
  });
}

export async function request(url, payload) {
  const response = await fetch(url, payload === undefined ? {} : {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-Palm-CSRF': window.PALM_BOOT.csrf },
    body: JSON.stringify(payload),
  });
  const result = await response.json();
  if (!response.ok || result.ok === false) {
    const error = new Error(result.error || 'Không thể lưu dữ liệu. Thử lại.');
    error.status = response.status;
    throw error;
  }
  return result;
}
