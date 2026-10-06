import test from 'node:test';
import assert from 'node:assert/strict';
import { sampleStroke, swipeSide, withPoints, nextPending } from './domain.js';

test('short gestures never choose a hand; both directions mean saving', () => {
  assert.equal(swipeSide(95), null);
  assert.equal(swipeSide(-95), null);
  assert.equal(swipeSide(-96), 'left');
  assert.equal(swipeSide(96), 'right');
});

test('freehand sampling preserves endpoints and image aspect ratio', () => {
  const points = sampleStroke([[0,0],[.5,0],[.5,1]],201,101);
  assert.equal(points.length,6);
  assert.deepEqual(points[0],[0,0]);
  assert.deepEqual(points.at(-1),[.5,1]);
  assert.deepEqual(points[1],[.2,0]);
  assert.deepEqual(points[3],[.5,.2]);
  assert.equal(sampleStroke([[.1,.1],[.11,.1]],100,100),null);
});

test('missing and unfinished lines remain unknown, never absent', () => {
  const original = { lines: { heart_line: { status:'unreviewed', points:[] } }, palm_width_points:[] };
  const partial = withPoints(original,'heart_line',[[.2,.3]]);
  assert.equal(partial.lines.heart_line.status,'unreviewed');
  assert.deepEqual(original.lines.heart_line.points,[]);
  assert.equal(withPoints(partial,'heart_line',[]).lines.heart_line.status,'unreviewed');
});

test('advance skips reviewed images and stops when the deck is complete', () => {
  const images = [{status:'pending'},{status:'approved'},{status:'pending'}];
  assert.equal(nextPending(images,0),2);
  assert.equal(nextPending(images,2),0);
  assert.equal(nextPending([{status:'approved'},{status:'rejected'}],0),-1);
});
