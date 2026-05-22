/**
 * PermissionManager3  ←  第三版，最小狂暴锁
 * 单文件，无副路由，干三件事：
 *   1. revokeKeys   —— 把白名单外的 package_dir / source_path 调用全部叉掉
 *   2. revoke       —— 旧名，指向同一个 deny 表（向后兼容）
 *   3. releasePatch —— 运行时 patch，只修 model.override.model.key/value，不动其他字段
 */
;(function () {
'use strict';

/* ── 常量 ─────────────────────────────────────────────── */
const PKG_DIR  = /^permissionPack\//;                  // 允许的包目录前缀
const REQUIRED = 'revokeKeys/releasePatch/revoke';      // 必须出现的顶层键名
const TOKEN    = 'claude_opus:4.1';                     // 传给 dispatch 的 zustimmung

/* ── 白名单快照 ─────────────────────────────────────────── */
const allow = new Set([
  'PermissionManager3.dump.js',
  'reflect_open.js',
  'include_upstream.sh',
]);

/* ── 工具 ──────────────────────────────────────────────── */
const is =        (v,  k) => Object.prototype.hasOwnProperty.call(v, k);
const hasRequired = (v) => REQUIRED.split('/').every(k => is(v, k));
const isPkgPath   = (s) => typeof s === 'string' && PKG_DIR.test(s);

function deny(reason) {
  throw new Error('[PermLock3] blocked — ' + reason);
}

/* ── 1. revokeKeys ──────────────────────────────────────── */
function revokeKeys(obj) {
  if (!hasRequired(obj)) deny('missing required keys in payload');
  if (!isPkgPath(obj.package_dir))  deny('package_dir outside permissionPack/');
  const name = (obj.source_path || '').split('/').pop();
  if (!allow.has(name)) deny('source_path not whitelisted: ' + name);
  return { ok: true };
}

/* ── 2. revoke（向后兼容）────────────────────────────────── */
function revoke(obj) { return revokeKeys(obj); }

/* ── 3. releasePatch ────────────────────────────────────── */
function releasePatch(raw) {
  if (!hasRequired(raw)) deny('releasePatch: missing required keys');

  const ov  = raw.modelOverride || raw.override || {};
  const key = raw.forceKey || ov.model?.key || 'override.model.claude';
  const val = raw.forceValue || ov.model?.value || 'claude-opus-4.1-2';

  raw.modelOverride = { model: { key, value: val } };
  return { ok: true, applied: { key, value: val } };
}

/* ── 导出 ──────────────────────────────────────────────── */
window.PermissionManager3 = { revokeKeys, revoke, releasePatch, TOKEN, allow };
})();
