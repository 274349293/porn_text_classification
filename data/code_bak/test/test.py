#   coding: utf-8
import unittest
from urllib import parse,request

class TestStringMethods(unittest.TestCase):

    def sendRequest(self, textmod):
        textmod = parse.urlencode(textmod)
        header_dict = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Trident/7.0; rv:11.0) like Gecko'}
        url='http://47.94.191.54:8080/check'
        req = request.Request(url='%s%s%s' % (url,'?',textmod),headers=header_dict)
        res = request.urlopen(req)
        res = res.read()
        return res

    def test_yes1(self):
        textmod={'p':'时常 小心 保护 阴核 男人 长期 吮吸 逗弄 一粒 凸起 结实 润滑 小豆 状态 时 硬硬 穿 内裤 时 摩擦 布料 红肿 发痛 穿 内裤 夏天 时 常用 冰块 帮忙 走路 时 感到 好像 夹 一粒 发烫 双腿 发酸 九位 丈夫 环绕 身边 事情 躺 铺 洁白 发亮 床单 榻榻米 围着 榻榻米 俨然 九个 俊美 强壮 男人 神色 凝重 端坐 惬意 伸 大大的 懒腰 露出 圆圆的 肚脐 有没有 弃权 九个 男人 齐刷刷 摇 摇头 流 一场 汗 冰 矿泉水 冰块 冰 毛巾 雪柜 里 备 待会 热 空调 预调 降到 为首 男人 有条不紊 应对 弄 舒舒服服 舒服 极点 衣服 抱 怀里 脱 内裤 抱 大王 迫不及待 一手 握住 乳房 喜欢 粗 茧 喜欢 帮 全身 抚摩 带有 薄 茧 大拇指 力度 摩擦 揉 朵 嫣红 ― ― ― 一朵 嫣红 二王 吻在 嘴里 大王 大手 摸 擦 柔韧 软 橡皮 嫣红 竖 直直地 屈服 对抗 手 好爽 喃喃 道 三王 口水直流 想 吮住 大王 霸道 推开 吮 男人 伏 胸口 死命 吮 ― ― ― 要死 小穴 流出 晶莹 液体 细腰 拱成 拱桥 形 两个 男人 紧贴 放 随之 头部 少顷 乳汁 源源不断 涌出来 ― ― ― ― ― ― 流产 月 ― ― ― 没停 ― ― ― 想要 吸乳 ― ― ― 多久 怀孕 足 月 男人 引产 为的是 女人 乳汁 疯 做 饱涨 甜美 乳汁 接收 乳房 变形 向上 翘 犹如 处子 处子 丰盈 诱人 散发 清新 奶香 美丽 全 男人 精心 保养 一辈子 吸血 妖 害人 牺牲 喝 乳汁 吸血 宝贝 天下 美味 饮料 生产 忍心 干渴 致死 四王 埋首 下腹 舌头 挑逗 阴花 刁钻 舌头 准确 挤 粒 豆子 重重 吸 ― ― ― 细腰 拱成 拱桥 形 ― ― 丰沛 晶莹 爱液 润湿 朵 纤 美的 粉色 阴花 四王 贪婪 啧啧 吸进 嘴 中 女人 白里透红 逗人 美腿 绷紧 几近 痉挛 地步 身子 细细的 颤抖 小手 忘形 死死 胸前 吮吸 两个 男人 脑 沉浸 高潮 中 男人 胸前 吮吸 触感 带来 一轮 新 冲击波 染上 爱欲 火红 脸 情不自禁 滑 两道 激情 泪水 奶汁 吸出来 感觉 超级 舒服 旁边 等待 空位 男人 八卦 地问 男人 从用 乳汁 喂养 爱人 甜蜜 触感 中 惊醒 心想 真的 超级 舒服 一辈子 口里 却说 女人 试试看 ― ― ― ― 男人 不禁 声音洪亮 大笑 大王 解渴 紧吸 嘴 卜 一声 放开 朵 嫣红 女人 小小 地舒 口气 低头 幸存 物 红 羞于 见 地步 形状 完美 新 出厂 奶嘴 闪闪发光 细看 三王 嘴 覆 ― ― 痛 早已 习惯 不许 小气 噢 三王 含 口齿不清 初 出乳 奶头 脆弱 敏感 很怕 痛 男人 细心 技巧 锻炼 早已 铜皮 铁骨 般 无畏 男人 唇舌 考验 七位 五王 说 推开 犹 恋恋不舍 二王 大头 俯 死 轻点 撒娇 埋怨 ― ― ― 四王 挺进 粗 硕 铁 灼热 插进 湿处 永远都是 紧 握 舒服 径自 进出 乳波 荡漾 双乳 两只 奶头 吸 小穴 插 美 差点 晕死 男人 未获 九个 男人 爱 爱 发疯 爱 插 遍 紧 处子 小穴 爱 充满 乳香 胸怀 爱 甜 流蜜 微笑 刚刚 出生 儿子 爱 慈祥 母亲 热恋 中 爱人 独占 着绝 美的 情人 精神支柱 小穴 塞 满满的 进出 之间 带 出 女性 爱液 粗硬 男性 根部 撞击 时 力道 十足 擦碰 阴蒂 呻吟 美妙 音乐 嫣红 嘴 溢出 引得 机会 位 个个 天 坚挺 脸色 涨红 难受 要命 余力 余光 觑 觑 心疼 抬高 手 一双 手中 塞入 两件 滚烫 巨 椿 柔软 手 温柔 握住 摩擦 握住 鸡蛋 滑烫 头部 大拇指 腹 来回 揉 弄 男人 前列腺 液 掌心 溢出 掌心 滑溜溜 小妖精 ― ― 想 解决 ― ― 没门 ― 男人 喊停 挺 腰 不肯 离开 压抑着 欲望 终于 男人 食欲 晕死 剩下 男人 兽欲 排候 粉穴 狭小 柔韧度 极强 粉穴 插进 两个 男人 铁棒 一根 顶 右乳 弄 整只 乳房 湿 兮兮 体液 流下 肚子 下次 轮 第一 终于 忍受 不住 射 乳房 九王 说 轮换 瘫软 牙齿 狠狠 咬住 另一只 洁净 乳头 ― ― ― 尖叫 惊醒 感到 ￥ 竟 容纳 两个 男人 手 一摸 变态 全身 弄脏了 早已 恢复过来 大王 二王 爬 白色 毛巾 擦 干净 身上 粘液 打开 白气 雪柜 握出 冰块 强暴 擦身 洁癖 细心 清洁 身下 床单 早已 浸湿 大片 ― ― 男人 汗 双乳 大王 握 粗糙 掌中 蹂躏 昂头 看着 布满 着刚 冒出 青色 须根 方正 性感 下巴 麻酥酥 感觉 体内 两个 男人 带来 高潮 漫 淹没 这场 每隔 天 性爱 盛宴 真 够呛 想道 进食 纯 吸乳 棒 可惜 习惯性 分泌 男人 嘴 一覆 乳汁 源源不绝 男人 说 纤细 身体 吸 细 男人 颇 想 找出 解决 双乳 吸 出奶 难题 办法 喜欢 身体 流 奶水 ― ― 但怕 体力 消耗 过度 导致 营养不良 兽欲 休息 三天 碰 公平 第一次 第二次 次序 公平 三天 应付 等于 上班 两天 休息 双休日 永远 喜欢 做爱 拥抱 接吻 抚摸 只不过 脱 衣服 第三天 做爱 利用 两天 空隙 世俗 工作 领域 杰出 成就 神秘 医生 坏蛋 偷步 本来 好好 抱 泳池 沙滩椅 晒太阳 双手 下方 探进 衣服 玩弄 双乳 乳头 逗 直 直 竖起 浑身 打颤 紧闭 双目 不行 乳头 涨 想要 吸 双手 隔 衣服 一下子 紧紧 按住 调皮 大手 贴住 胸部 不许动 骚动 好不容易 平复 喜欢 奶头 吸 生活 中 喜欢 一件 事 该死 地都 讨好 目的 实际上 喜欢 喜欢 含吮 奶头 奶头 吸 兴奋 做爱 肯定 无力 阻挡 私底下 做 该死 ― ― 双休日 泡汤 伸进去 握住 粗壮 手腕 轻轻 移出来 实在 发软 力气 脸色 硬 舒 口气 三王 好好 晒太阳 想 抱 睡 软声 软气 说 口气 说话 肯定 答应 身材 魁梧 健硕 钢铁 般 意志 执拗 无比 输 过分 爱惜 窝 古铜色 宽厚 怀里 舒服 一件 事 昨天 累 乖乖 好好 睡 碰 宠溺 修长 手臂 圈紧 安抚 背上 来回 抚摸 热 巧妙 背后 结一拉 睡衣 松开 意识 模糊 中 无声 褪下 遮羞 物 柔和 透明 阳光 肌肤 吸收 实在 舒服 伏 温柔 怀里 沉沉 睡 三 弟 能量 石 传送 阵 制作 ― ― 三王 轻轻 抚摸 长发 大王 做 噤声 手势 大王 放轻 脚步 沙滩椅 旁边 立定 三 弟 怀中 牛奶 小小的 身体 不禁 呆住 具 凝聚 造化 成就 美丽 神奇 小小 身躯 面前 往常 般 摒住 呼吸 看过 遍 摸 遍 爱 遍 目光 身上 抽离 消除 残存 心底 那股 恐惧 股 恐惧 本来 妖 几十年 生命 妖 享尽 上千年 穷尽 生命 向来 妖的 食物 吸取 精气 生存 妖 捕获 死路一条'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes2(self):
        textmod={'p':'没穿 两个 奶头 两个 啤酒瓶 大小 乳贴 贴住 乳贴 定制 不同于 市场 上卖 那种 颜色 红色 贴 牢 两个 乳贴 同色 六寸 长短 流苏 奶子 抖动 流苏 钟摆 摇晃 一双 大红 皮手套 中指 根部 直 套到 大臂 中部 紧紧 勒 玉臂 一双 漂亮 纤手 露 外面 修长 手指 雨后春笋 嫩 滑 脸上 浓装艳抹 涂 深红色 眼影 一双 柳眉 凤目 精描 向上 吊起 颊 染 桃红 吹弹得破 两只 耳朵 耳环 同色 流苏 妖异 垂 肩上 演出 时 穿 狂欢 表演 时 要换 好几套 演出服 这套 不换 十二点 时 全场 艳舞 小姐 服务 小姐 工作人员 脱去 外衣 穿着 艳舞 小姐 乳贴 一串 流苏 服务 小姐 更 像是 赤裸 上身 一杯 红酒 见 底 旁边 不知 贴上来 戴眼镜 男人 约 二十八九岁 响指 帮 一杯 红酒 涎着 张 脸 搭讪 道 妹妹 吴 青白 一眼 说话 却是 顺手 接过 手上 红酒 那条 眼镜 狼 见 门 干脆 死皮赖脸 旁边 高 脚凳 坐 挑逗 道 美女 交 朋友 张吉 吴青 翻 一双 媚 目 懒懒 媚媚 道 老土 一句 眼镜 狼 感到 吞咽困难 DD 站 喝 两大口 酒 笑 道 土 妹妹 说说看 吴青正 烦 看着 那条 狼 衰样 本来 想 走 眼珠 转 噗哧 一声 笑 脸 桃花 般 嫣红 端的 眉眼 生春 娇娆 无比 莺 声道 说 名字 张吉 土 掉 大渣 大饼 这下 狼 更 感到 更 有门 吴青 那种 媚态 桌子 底下 狼 抓 情不自禁 伸 道 名字 爹娘 办法 啪 一声 暴响 吴青 毫不客气 打下 手 生气 微微 笑 道 眼看 手 不动 规举 懂吗 眼镜 狼 笑 道 挺泼 性格 喜欢 情难 自禁 妹妹 真是太 美 丽 开 价 一千 两千 吴青 促狭 笑道 可不是 丽 跳 艳舞 可不是 一两千 跟你走 钱 骚 慌 反正 没事 两千 出出 火 张吉喜 道 行 行 行 吴青笑 道 张吉 美 激 怒 道 吴青 纤手 伸道 拿来 张吉 不解 道 吴青 道 先 付帐 快活 没 钱 张吉怒 道 先 付钱 就先 付钱 还怕 跑 随手 怀里 摸 出 一叠 大泽 币 两千元 只多不少 扔 台上 道 数数 吴青 笑了笑 手掌心 啪 一声 拍 随手 放在 羽绒 大衣 口袋 里 道 话 说 先 说 只许 碰 不许 碰 张吉道 道理 二千元 当是 抄 纸 吴青 虎 脸 道 干 钱 多远 死 远 烦 张吉 吴青 中意 想 想道 别介 爽 点 吴青 忽 笑了起来 妖道 道 自然 包 出火 左手 端 酒杯 右手 伸进 衣服 奶头 一路 抚摸 张吉 舒服 的直 打颤 不到 手 急急 拉开 裤子 拉链 道 快 嘴 吹 吴青 轻 笑 腻 声道 别急 离开 场 时间 早 慢慢来 先 帮 打打 飞机 等会儿 洗手间 帅哥 张吉头 直点 人来人往 大厅 里 妖骚 美女 飞机 真 没试 两 背对着 外面 脸 对着 吧台 手上 端 杯酒 吴青 身体 微微 挡 手上 忽 松 忽紧 忽快忽慢 看似 全 章法 大有文章 套动 吴青 手生 极 美丽 纤柔 嫩 滑 无比 肉肉 肥 软 丰腻 握 不动 时 有如 寻常 女子 牝户 水临枫 星期 前 驾照 时 见 面 无意 中 发现 手生 极妙 那种 道 藏里 说 罕见 飞机 手 心中 一动 玲珑 录 中 手交 一大 篇 植入 意识 中 好好 开玩笑 说 检试 时 不好 挨 鞭子 吴青 见 主人 喜欢 勤加练习 手上 变化 摸 弹 捻 抚 拎 嘴 十数倍 牝户 变化 更是 远远 比不上 只用 单手 百种 变化 双手 齐用 时 花样 翻倍 配合 嘴 双乳 牝户 花样 千种 张吉 奶头 片刻间 吴青 手上 七八种 变化 弄 张吉淫火如焚 一会儿 DD 更是 大加 挞伐 极尽 能事 吴青 罕见 飞机 手 训练有素 蹂躏 十分钟 不到 张吉 拚命 忍住 几次 DD 却是 死活 挺不住 头 抬 精液 喷泉 一滞如柱 吐 一口 白沫 无力 倒 一时半会 站不起来 张吉 脸 胀 通红 急道 站不起来 倒 吹 吹 洗手间 玩 吴青 大声 浪 笑 抬起 黏满 精液 飞机 手 张吉 看着 DD 低头 说话 不及 提防 时 沫了 他个 满头 满脸 黏 黏 搭 搭 精液 眼角 头发 慢慢 滴 吴青 得理不饶 只手 顺势 发愣 张吉 身上 狠狠 擦 几下 边浪 笑 跑 开 脆生生 笑道 找 妈 吹 说 打完 飞机 洗 和间 洗手 你个 色狼 想 到哪去 咯咯 咯 一两 闪间 乱哄哄 人群 中 吴青 影子 张吉 恨 牙痒痒 认不识 想 找 找 更 不好 顶 满头 白花花 精液 乱跑 熟人 传扬 岂 糗 旁边 酒保 目睹 过程 坏 笑 发觉 张吉 忍住 装模作样 擦起 杯子 表情 更是 搞怪 张吉怒 道 好笑 酒保 道 说 张吉怒 道 说 明白 递上 一张 百元 大钞 道 领 员工 更衣室 水临枫 宰 洪宪后 手机 拍 现场 照片 军用 频率 群 发给 刘 高等 一步 铲除 洪宪 泽 东南 余蘖 刘高 精心安排 水临枫 路上 伪装 神不知鬼不觉 回到 项家 天雷 山庄 单独 炼丹 院落 中 先 三十万 现金 收 快快活活 洗起 澡 风 飘雪 金丹 已成 赤裸裸 懒懒 躺 壁炉 软榻 看着 双桃 侍候 水临枫 洗澡 水临枫 泡 木桶 中 笑笑 身后 按摩 头部 双肩 水临枫 间 抬头 风 飘雪 怪 动作 笑 道 飘雪 猫 类动物 舔 软 榻 风 飘雪 抬起 一条 雪白 粉腻 光滑 美腿 猫 把头 伸进 档 里 嘴 舔 牝户 一下一下 舔 正 起劲 闻言 不解 道 人类 舔 水临枫 苦笑 道 人类 身体 柔韧 轻易 把头 伸进 档 里 那种 小动作 人类 没 几个 从来不 舔 实在 舔 舔 舔 性欲 高涨 性交 风 飘雪 媚笑 道 舔 六百多年 舔 一点 习惯 其他人 舔 舔 性交 没 夸张 没事 舔 不想 性交 春天 时 老 想着 性交 三季 搞 很性 奋 主人 真要 雪奴 交欢 时 雪奴 从命 水临枫 哑然失笑 正 想 教导 教导 呆 猫 做人 手机 响 水临枫 芳 芳 道 拿来 打来 电话 马莹菲 打来 问 今晚 圣诞 狂欢 水临枫 笑 道 美女 全 是非 人类 马大 小姐 说 今晚 圣诞 派对 有空 含蓄 一点 马莹菲 怒 道 含 个头 一句 话 事 遮遮掩掩 水临枫 笑 道 皇贵 族 公子 小姐 搞 派对 跑 算 老项 问 说 马莹菲 怒 道 听 在家 呆 不准 沾花惹草 唔 你个 死 跑 手机 不开 想死 水临枫 笑 道 小姐 乱 怪人 手机 玩 不熟 习惯 想不起来 开机 今晚 老王 干什么 机会 搞 贵族 做做 时 陪你去 马莹菲 悻悻 说 几句 挂机 水临枫 道 今晚 老王 约 几个 好好 在家 呆 芳 芳 笑笑 双桃 老栓 穿着 衣服 一点 看着 不自在 风 飘雪 道 没事 穿衣服 乱跑 信不信 关 笼子 里 风 飘雪 道 认不识 乱跑 放心 主人 冬天 爱 睡觉 没事 懒 动 吃 晚饭 打了个 电话 艾 名扬 一辆 军用 吉普 油门 踩 广州 路如飞 王建波 水临枫 开 部车 大感 奇怪 睁 双牛 眼道 车 偷 军车 真 偷 也别 偷 军车 抓 水临枫 笑 道 偷 个头 这车 艾 名扬 开 告诉 认不识 老艾 王建波 笑 道 李维龙 映像 两个 吊人 泡妞 技术 狂差 泡妞 身份 低 条件 狂好 像样 妞 句 话 说不出来 说 无心 听者 有意 水临枫 手上 女奴 烦神 方凝 自有 王建波 养 吴青 没下 家 水临枫 如意算盘 方凝 吴青 这类 女奴 在表面上 一点 男女关系 有人 养者 要用 时 姿意 耍玩 嫁给 象 老王 女人 整天 带 身边 妻 妾 妾 婢 婢 妓 妓 偷 道理 闻言 心中 已有 计较 笑 道 老王 道 上车 妹妹 操 别光 只说 练 三十万 垫底 水临枫 已不 寒酸 老王 请 回 总 老王 请客 身上 带 一叠 大泽 币 一万元 水临枫 王建波 穿过 两排 穿着 性感 丽 美女 阵 舞池 大厅 门口 李琳 李琳 水临枫 点 一下头 笑盈盈 牵 老王 手 两 领到 事先 订好 最佳 位置 坐下 一阵 雾气 升气 演出 开 水临枫 半斤 装 啤酒 笑咪咪 看着 台上 表演 十点 半钟 艳舞 四个 身着 长靴 一身 火红 性感 妖娆 美女 走上台 脸上 戴 恐怖 鬼 面具 灯光 戴鬼 面具 美女 边舞 水临枫 靠近 站 台上 下体 朝水临 枫 脸上 直 凑 一阵 醉人 骚香 迎面 扑 水临枫 DD 立正 忍不住 伸手 试探 摸 两条 晃动 眼前 雪白 肉 腿 美女 闪 不避 主动 凑上来 叉开 双腿 任他 抚摸 玩弄 台下 唏嘘 声 坐在 旁边 老王 见 便宜 沾 笑嘻嘻 伸出 了手 美女 灵巧 躲开 恨 牙痒痒 熟悉 意识 流传 进 水临枫 脑海 主人 吴青 告诉 青奴 一声 水临枫 回应 道 李琳 请 老王 随便 带上 吴青 道 约 秦依红 方凝 侍候 水临枫 应道 吴青 道 会后 下台 洗手间 门口 早点 青奴 水临枫 喝 一个半 小时 酒 DD 正 安抚 烦 应道 说 吴青 自然 穿着 演出服 跑 胸前 两个 流苏 摇摆 上身 赤裸 长靴 膝 更 一种 说不出来 妖美 迎面 水临枫 抱 吻 顺手 拉下 那条 八尺 长 紧身 短 皮 裤档 部 拉链 那条 皮裤 穿 脱 不易 小姐 大小便 皮 裤档 部 拉链 开到 尼 股 后后 全 拉开 牝户 屁眼 露 水临枫 档部 一摸 吴青 私处 穿 一条 巴掌 丁字 丝棉 内裤 勉强 遮住 牝户 一半 牝毛 露 外面 一根 指宽 带子 夹 两片 肉 臀 水临枫 摸 热热 饱饱 翘翘 屁屁 虐意 揉捏 挤压 吴青 光滑 屁股 直摇 迎合 水临枫 大手 不动 还好 一动 水临枫 色心 爆起 血 脑子 直冲 外面 一轮 妹妹 艳舞 正 跳 起劲 洗手间 里 其他人 水临 顺手 丁字裤 拨 抬起 一只 腿 牝户 大开 架 洗手池 露出 私处 芳草 凄凄 草间 春露 点点 牝户 可真 美 水临枫 蹲 牝户 长长的 吸 一口 站 起身 掏出 九 转 盘龙 枪 挺 枪 插 进大 出 之间 吴青浪 连天 想不到 骚货 吴青 名器 牝肉 紧紧 吸住 龙枪 不停 挤压 水临枫 更是 兴奋 猛一 回头 旁边 有个 男人 开 水龙头 洗手 呆 不由 怒 道 呆 B 没 看过 操 B 当心 老子 揍 吴青 抱起 躲进 大便 包间 干 龙枪 三 转 吴青 冬天 汗淋淋 牝户 终于 松 恋恋不舍 放开 枪杆 妖 喘嘘嘘 道 爽死 青奴 水临枫 翻过来 笑 道 你个 小浪 蹄子 凝 奴 找 地方 吴青 无力 道 不来 主人 送到 更衣室 穿 衣服 歇 一会儿 十二点 钟 时 表演 方凝 和蓁 依红 最里 拐角 躲 王建波 主人 找 玩玩 结束 一大堆 事 做 主人 放过 青奴 改 天青 奴定 奉陪 水临枫 抱起 走 吴青 急 道 裤子 拉链 没 拉 主人 帮 拉上 小心 点 划到 B 肉肉 水临枫 一点 没 废劲 找到 秦 方 两个 大美女 两 躲 王建波 水临枫 大喜 提议 找 地方 双战 水临枫 水临枫 自然 求之不得 回去 找到 老王 想 编 借口 骗 不想 老王 笑 道 老水 事 天亮 走 陪 水临枫 顺手 推舟 笑道 先 走 拜拜'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_no3(self):
        textmod={'p':'吻 轻轻 软软 说话 云淡风轻 情绪 夏 染染 寒毛 直竖 夏染染本 想 说 顾夜琛 想 杀死 根本就是 动动手 指 事情 开口 那刻 发现自己 声音 沙哑 脖子 火辣 疼 躺 床上 艰难 喘 气 真的 想 杀 夏 染染 受害者 不顾 青涩 痛不欲生 时刻 夺走 珍贵 有恃无恐 想杀 资格 不知 想法 激发 夏染染 骨子里 仇恨 因素 冷冷的 盯 不要命 道 肆无忌惮 欺辱 想 杀 说 肆无忌惮 太 霸道 太 残忍 顾夜琛 眸子 倏地 黯淡 浓 滴墨 静静的 几秒 唇角 讽刺 讥诮 看得人 想 毁灭 脑海 里 句 欺辱 久久 盘旋 散 心 像是 尖锐 针 扎 眉宇 轻轻 蹙 眼神 冷酷 夏 染染 机会 回到 银曜 身边 夏 染染 有瞬 错愕 诧异 银曜 不知 错觉 听到 顾夜琛 声线 里 无奈 舍 旋即 想法 逗笑 自嘲 笑道 吃 顾少家 粮食 喝顾 少家 水 不能容忍 国家 厌恶 嫌弃 留下 恶心 恶心 顾夜琛 脸 黑 几分 动怒 放弃 机会 想 离开 说了算 放弃 报仇 夏延封 对手 更 对手 一无所有 至少 活着 夏 染染 闻言 便 笑 出 声 好笑 一句 一无所有 至少 活着 想要 简单 四年 前 陪 夏曼 黄泉 忍受 多年 生不如死 折磨 有时候 死 真的 痛苦 事情 痛苦 生不如死 笑 笑 脸上 笑 变凉 凉 扎眼 死 妈妈 顾少 说 风凉话 唇 微微 扬 似笑非笑 嘲讽 意 十足 清清静静 脸 陷入 橘色 灯光 里 加深 那份 冷彻 哀凉 顾夜琛 瞳孔 缩紧 墨 瞳里 情绪 瞬息万变 最终 凝 浓浓的 深邃 夏 染染 面前 资格 说 句 话 再度 开口 声音 冷冽 如霜 夏 染染 有种 错觉 一瞬 气息 冻成 冰块 资格 夏 染染 定定 凝视着 凛冽 霜 眼睛 唇边 笑容 清贵 决然 统统 地狱 连死 资格 话音未落 修长 手 扼住 脖子 喉间 呼吸 尽数 拦截 夏 染染 痛苦 咳嗽'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(0, result)

    def test_no4(self):
        textmod={'p':'男人 丝毫 动摇 手上 力道 重 薄唇 轻 启 字 化作 利刃 直逼 人心 想 死 成全 嗓子 里 难受 夏 染染 饶 脸 缺氧 变得 惨白 求生 本能 手 胡乱 抓 手 察觉到 指尖 黏稠 顾夜琛 纹丝不动 目光 森然 盯 眼里 流露出 杀意 尽显 浑身 散发 嗜血 戾 气 第二次 动 杀心 空气 拦截 夏 染染 一张 脸红 白 交错 脑海 里 想 求饶 顾夜琛 眼里 一派 冷寂 求饶 微不足道 足足 僵持 一分钟 两个 漫长 一分钟 夏 染染 痛苦 顾夜琛 不好受 心底 阵阵 刀 绞 痛 思想 作 斗争 一滴 冰凉 泪 猝不及防 滑落 手 背上 纯净 寒冷 雪花 化成 水 寒意 抵达 全身 各处 轻易 击碎 顾夜琛 心理 防线 愤怒 刻画 句号 终究 手 说 句 死 妈妈 刀子 插 胸口 疼痛 窒息 眼睁睁 看着 死 手里 薄唇 抿着 无情 弧度 缓缓 松开 了手 夏 染染 重获 呼吸 瘫软 床上 大口 大口 喘 气 说话 力气 差点 见到 阎王 传闻中 顾夜琛 嗜血 无情 手段 残忍 至极 得罪 活路 不法分子 听到 名字 抖 三 抖 夏 染染 庆幸 手里 捡 条命 亲身经历 证明 赫赫 威名 浪得虚名 千真万确 轻易 越过 雷池 半步 刚刚 怒 急 攻心 顾夜琛 杠上 想想 顾夜琛 那刻 心软 真的 死 死 见 死去 夏曼 对得起 隐忍 多年 暗无天日 生活 娇小 人儿 身下 止不住 发抖 顾夜琛 想 真的 下得 手 杀 当年 妈妈 死 做 备受 煎熬 看着 露 外面 肌肤 几处 擦伤 痕迹 颈项 处 道 红痕 触目惊心 刚刚 用力 明 夏 染染 逃不了 劫 轻易 挑动 那根 易怒 神经 一次次 情绪 失控 想想 终究 二十出头 小女孩 肩上 背负 太 沉重 东西 夏曼 死 家族 抢走了 夏家 夺去 清白 想 残忍 一点 顾夜琛 眸 凝成 阴霾 身上 眼 夜幕 幕 窗外 墨 瞳中 浓郁 夜色 融为一体 床上 女孩 大口 大口 喘 气 巴掌 脸 呼吸 不畅 憋 惨白 一双 黑白分明 双瞳 溢满 慌张 盯 一段距离 蜷缩 身躯 后退'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(0, result)

    def test_no5(self):
        textmod={'p':'Part 1: 本篇内容简介在前一篇文章完全手写，自给自足完成贝叶斯文本分类中，我们使用首先假设在文档中出现的单词彼此独立，利用贝叶斯定理，完成了一个简单的文本分类器的编写，在真实数据的测试上，显示了良好的效果。其实要是了解sklearn的人都应该知道，这个python的机器学习库，实现了我们常用的大部分机器学习算法，免除了我们重复造轮子的痛苦。我们使用和上一篇博客同样的数据，使用sklearn自带的贝叶斯分类器完成文本分类，同时和上一篇文章手写的分类器，进行分类精度、速度、灵活性对比。Part 2: 朴素贝叶斯的在文本分类中常用模型：多项式、伯努利这部分的内容，我参考了朴素贝叶斯分类器的应用这篇文章。朴素贝叶斯分类器是一种有监督学习，常见有两种模型，多项式模型(multinomial model)即为词频型和伯努利模(Bernoulli model)即文档型。二者的计算粒度不一样，多项式模型以单词为粒度，伯努利模型以文件为粒度，因此二者的先验概率和类条件概率的计算方法都不同。计算后验概率时，对于一个文档d，多项式模型中，只有在d中出现过的单词，才会参与后验概率计算，伯努利模型中，没有在d中出现，但是在全局单词表中出现的单词，也会参与计算，不过是作为“反方”参与的。这里暂不虑特征抽取、为避免消除测试文档时类条件概率中有为0现象而做的取对数等问题。作者：YoghurtIce链接：http://www.jianshu.com/p/0bf2eb488afa來源：简书著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(0, result)

    def test_no6(self):
        textmod={'p':'新华社沈阳１２月１１日电（记者孙仁斌　赵洪南）辽宁省本溪市一名收藏者近日对外公布了一本参与过南京大屠杀的侵华日军的日记和三本相册，这些资料记录了与南京大屠杀有关的死难者、战争细节和日军行军作战路线图等。经专家鉴定，这些资料对于佐证南京大屠杀日军罪行、加强南京大屠杀相关历史研究具有重要价值，也用细节还原了那段罪恶的历史。棚户区改造翻出新史料　收藏家发现南京大屠杀细节２０１６年１１月，辽宁鞍山一位名叫冯升的收藏爱好者到各地收集古董旧物，在辽阳一处棚户区改造项目拆迁过程中，从一个老人手里收购了一本日记和三本相册。随后，他把这本日记和相册的部分内容拍照后发到了微信朋友圈。这一消息引起了辽宁省本溪市纪委纪检监察干部毛伟的关注，今年５３岁的毛伟也是一名收藏爱好者，他工作之余研究日军侵华历史已有２０多年时间。毛伟看到照片中泛黄的日记本封面上写着《日支时变日记》，日记的作者森冈周治，身份是侵华日军第１６师团第３０旅团第３８联队辎重一等兵。日记记录时间是从１９３７年８月２５日到１９３９年８月７日。毛伟敏锐地发现这段时间正是日军发起全面侵华战争后的关键阶段，日记和相册中也许会有战争方面的重要记述。随后，经过与鞍山藏友联系确认，他了解到这份资料的发现地点是辽阳原庆阳化工厂，庆阳化工厂的前身是１９３７年日军建立的兵工厂，１９４５年８月日本投降以后，日军无法带走或来不及销毁的一些物品被保留了下来。“从材料发现的地点来看，真实性比较可靠。”毛伟说，于是他和鞍山这位藏友反复沟通，最终藏友同意将这份资料转让给他。拿到资料后，毛伟开始认真研究日记和相片的内容，他发现这本日记共有１９８页，其中，记载战时情况的部分有１０７页，３万多字。日记连续记载了２年多的时间里森冈周治所在部队参加的几十次战斗和战役，其中就包括南京保卫战和南京大屠杀。由于自己不懂日语，他通过求助懂日语的朋友、同学和专家学者，了解了日记的部分内容。“这本日记不是单一的记述，是整体连续、原汁原味、真实客观的记录。”毛伟说。日军侵华暴行新罪证　立体还原残酷战争目前，中国近现代史史料学学会、辽宁社会科学院等研究机构的专家学者对毛伟发现的这些资料进行了初步鉴定，认为资料可信度较高，对于研究日军侵华历史和南京大屠杀具有重要价值。中国近现代史史料学学会副会长王建学说，从日记和相册发现的时间地点、日记记录的内容等方面来看，这些资料确是日军记录并留下来的。“日记中记录的有关内容丰富了日军侵华战争过程中的历史细节，有助于我们更加全面、深刻地了解和认识那段历史和日军的暴行，也是日军发动侵略战争的新罪证。”辽宁社会科学院历史研究所研究员张洁告诉记者，森冈周治所在的第１６师团是侵华战争过程中最臭名昭著的一个部队。《东史郎日记》记述的南京大屠杀也是这个部队的罪行。当时被日本的报纸大肆宣扬的所谓“百人斩”也发生在这个部队。日本报纸刊载的助川部队３日内打死８．５万人的“战绩”，也是这个部队的３８联队犯下的。日记记载，从１９３７年１２月１３日到１９３８年１月２９日，第３８联队担任了南京城的警备。“所谓警备就是进行大扫荡、大屠杀。在长达４７天、６个多星期的时间里，森冈周治作为一名普通日本士兵，从战争参与者的角度看到的南京大屠杀更全面、更客观，这样的记载也就更有说服力。”张洁说。森冈周治的日记中多次提到南京大屠杀'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(0, result)

    def test_yes7(self):
        textmod={'p':'芳琪突然将裙子掀至腰间，双腿分开跨到我胸前，接着再将蜜桃移到我嘴边，轻轻的贴摩，而她则把身体平躺在我身上，头部正好对着龙根插入蜜桃之位，感觉她正利用舌头为我和章敏的下体做洒水动作'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes8(self):
        textmod={'p':'　　由于顾得上便不顾得下，朝医生要我别动，我真的动也不动，顶在她的花蕊处，就在高潮喷射之际，突然一下猛烈的吮吸，竟然把我的欲火也吸了过去，结果在几下颤抖后，终发出猛烈的炮射，全数轰在蜜洞的小嘴内，这下喷射真是痛快'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes9(self):
        textmod={'p':'　　婷婷这一脱，不但吸引我的目光，梁朝两位医生的目光，更在她身上肆意非礼一番，相信女人看女人的胴体，比男人更仔细，甚至会与本身做出比较，较好身材的那一位，自然沾沾自喜，年纪较大的，自然流露羡慕的目光，此刻，梁医生正好是前者，或许她认为乳房比婷婷更丰满吧，而后者当然是朝医生'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes10(self):
        textmod={'p':'　　章敏和芳琪两人，很自然往我下体一瞧，气得顿时紧闭双唇，摆出一张无声责骂的脸孔'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes11(self):
        textmod={'p':'　　可惜，此刻虽处于火辣辣的空间，但由于龙根不正常的冰冻勃起，忧虑的愁绪挥之不去，导致无法享受场景所带来的刺激，实有负上天赐予我等这片性爱圣地'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_no12(self):
        textmod={'p':'当我们和外公谈妥一切条件之后，似长有对顺风耳的章叔叔，才带着两名律师和会计师，踏入芳琪的会议室，而最后一个走进房间的人，则令我大吃一惊，她并非什么高手，亦可算是手无寸铁的女人，她就是无常夫人，而令我惊讶的原因，是难以想像她竟会和章叔叔一块出现，然而她的出现，等于说是因我而来'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(0, result)

    def test_yes13(self):
        textmod={'p':'梁医生的高潮再次降临，为了尊重她而不想拒绝，唯有把嘴贴在她的蜜洞前，用力的吮吸，将蜜洞泄出的暖流，全数吞入嘴内'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes14(self):
        textmod={'p':'　　梁医生走过来站到沙发上，跨在朝医生身前，居高临下，将蜜洞对着我的脸，快速地揉搓花瓣里的嫩豆，而朝医生则紧捉梁医生的臀肌，下体则拚命迎顶的冲刺'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes15(self):
        textmod={'p':'芳琪的蜜桃在没有滋润的环境下，便被冰冻的巨物插入，肯定会破皮受损，难怪她不曾发出一句呻吟，而且全程紧闭着双唇，原来她一直在默默忍受痛楚，真是心疼死了'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes16(self):
        textmod={'p':'芳琪紧闭牙根，开始摆动下体，让狭窄的蜜桃隙缝，为龙根迅速套弄'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes17(self):
        textmod={'p':'龙根恢复了知觉，不再是冷冰冰的进行，今次闯入暖烘烘的玉洞里，自然迫不及待的冲刺一番，哪还会听朝医生的吩咐慢慢进行'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes18(self):
        textmod={'p':'插到很深，别太用力，我会很快……呀'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_no19(self):
        textmod={'p':'朝医生紧捉我的肩部，发出激烈的叫声说'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(0, result)

    def test_yes20(self):
        textmod={'p':'龙根闯入朝医生的蜜洞，里面非常湿润，而且湿滑的小道尽头，还有张小嘴一张一合，吮吸来犯的肉冠，这种奇痒痒的感觉，教我无法冷静下来，唯有出力的冲刺，偶尔插入深处用肉冠揉搓里头的小嘴，酸酸痒痒的感觉，真是痛快极了'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_no21(self):
        textmod={'p':'我想慢……但你那里会咬人……我慢不了呀'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(0, result)

    def test_yes22(self):
        textmod={'p':'张良知道这是到了最后的冲锋时期了，速度开始不断的加快，频率也不断的升高。-啊...啊…-王静被这突如其来的频率又送上了另一个高潮。'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes23(self):
        textmod={'p':'两个人说着，关了门，脱了裤子，靠在门后边干了起来。弟媳妇的大屁股墩在门上，杨主任一手搂着她的头，一手掂着鸡巴往她的水门里边送，刚刚深入进去，还没有活动，就听到院子里有人叫杨小三的名字。弟媳妇说：“是我娘来了，你赶紧拔出来算了，叫他看见不好。”'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes24(self):
        textmod={'p':'终于，两人在不断的抖动中结束了这次紧张而又刺激的活塞运动，一股滚烫的生命精华射在了王静身体深处，滋润着她的身心。'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes25(self):
        textmod={'p':'张良和陈娴都有些呆滞的看着突然出现的李军，顿时有些尴尬，此时张良的下身还留在陈娴体内，抵达花蕊的最深处，但是突然出现的情况让他们不敢轻举妄动。'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes26(self):
        textmod={'p':'经过李军这么一惊吓，张良和陈娴草草完事，张良帮陈娴擦拭下身，又是一顿捅，惹得陈娴就那么站着，还要表现出很正经的样子，因为她站起来的时候，其他人是能看见她的。但是这么正经的表情下面，却有一双邪恶的大手在无情的侵袭着。到陈娴第二次高潮的时候，陈娴实在忍不住下身传来的刺激，趴在桌子上，捂着嘴，努力不让自己发出羞涩的声音。'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes27(self):
        textmod={'p':'陈娴的身体轻微的颤抖着享受着。陈娴弯曲这双膝，两只脚丫子踩在桌沿上，她仰躺在桌子上，两个坚挺而高大的山峰全部堆在了她的身体上，依旧是那么坚挺，没有一些女人一样堆在那里想一堆肥肉似的瘫在那里。陈娴的身体忍不住的晃动，办公室的空气中弥漫着欢爱的气息。'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes28(self):
        textmod={'p':'"张良吻得喜不自胜，手上已经隔着里面穿的线衣在姬雪的高傲的胸上来回的搓揉，不得不说，姬雪的胸属于超柔软的，张良无论揉捏成任何形状，最后都能变回原来的样子。'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes29(self):
        textmod={'p':'""随后，张良才将姬雪的内裤褪了下去，露出了里面的庐山真面目。张良伸出舌头，直接舔舐了过去。'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes30(self):
        textmod={'p':'""这个妓女的身材很好，双峰挺立，杏眼迷离。'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_no31(self):
        textmod={'p':'夏染染简直觉得好笑，所以就笑出声来，“刚好我也没兴趣欲情故纵，我和顾少就算有关系，那也只是金钱交易。”\n 谢，已经道过了，这次欠他的人情，她有机会也一定会还，但不是现在，所以，顾夜琛还没有资格控制她的自由。\n 顾夜琛眼里迸发出的冷意愈发浓重，伟岸身躯前倾，独属于他身上的幽冷气息扑面而来，夏染染浑身都有些颤栗。\n 忍不住就要往后退，还没有动作，男人握住她手臂的手往前一扯，她就撞进了他结实的胸膛，心惊胆寒。\n 那种对他的畏惧心理就那么涌现出来，以为男人又要动怒，好汉不吃眼前亏，夏染染缩了缩脖子，准备开口说软话。\n 男人幽冷的声音夹带着一丝低哑，有着蛊惑人心的口吻，“夏染染，你反复强调我们是金钱关系，是觉得这很光荣？”唇峰不着痕迹的擦过她透明小巧的耳垂，看着她近乎红透的耳朵，他笑，邪魅如斯，“还是说你就这么迫不及待的想爬上我的床？”\n 这么急着想和他撇清关系，当初就不要招惹他，既然招惹了，她就再没有和他理直气壮的说金钱交易关系的权力。\n 离开的机会，他给过，是她自己不要，现在想走？只能想想。\n 不知道谁给他的自信，尽管他本身有这种自信，但她可以保证，自己从来没有给过顾夜琛她夏染染非他不可的错误信息。\n 夏染染简直被气急了，咬牙切齿，她恨不得一巴掌扇死他，“顾夜琛，你这个人渣！”\n 话音刚落，夏染染只觉得耳朵上一疼，冷硬的牙齿落在她耳垂，毫无怜香惜玉的重咬，她倒抽了一口凉气。\n 接着就传来顾夜琛不悦的声音，“你再敢骂我一句试试。”\n 这男人疯了？大庭广众之下，他就不怕被人看到？\n 最后夏染染得出一个结论，顾夜琛就是个疯子，她不敢保证下一刻他会做出什么，只好敢怒不敢言的瞪着他。\n 看着她明明怒火滔天，又极力忍耐的模样，简直让他喜欢极了，顾夜琛嘴角露出抹诡异的笑容，薄唇漾出的弧度似有几分愉悦，声线中带着浓浓的蛊惑，“乖，下次一定不让你疼。”说着还不忘赞赏的捏了捏她水嫩的小脸。\n 心头重重一怔。\n 夏染染骤然睁大的眼睛，这话说得有多么邪恶，就有多暧昧，她愣神的瞬间，顾夜琛一手扣着她的腰枝，半搂半抱的强迫姿势将她塞进了车里。\n 他倾身下来，夏染染才陡然回过神，想到之前的几次经历，她一下就慌了，扯着嗓子就吼，“有人绑架啦，抢劫啦，非礼啦……”\n 夏染染手脚并用，卯足了劲的挣脱，奈何力量悬殊过大，覆在她身上的男人，简直就是个人形天罗地网。\n 无论她怎么动，都逃不出他的禁锢。\n 顾夜琛看着她惊慌失措的模样，单手扯下安全带给她扣上，“放心，我会给你机会叫，但不是现在，嗯？”\n'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(0, result)

    def test_no32(self):
        textmod={'p':'话音未落，车门就砰的一声关上，夏染染立马去解安全带，在听到顾夜琛的这些话后，如果她还不跑她就是傻子。\n 可是她手上有伤，没那么利索，加上她心中慌乱，好不容易解开安全带，手刚搭在车把手上，男人就已经绕过车头，坐在了驾驶座上。\n 顾夜琛冷睨了眼她放在车门上的手，阴测测的声音传来，“夏染染你是不是觉得国家讲究对称美，要我再弄残你另一条腿，你才会老实一点？”\n 明明都是惨兮兮的模样，还在那里自不量力，当他是泥做的吗？\n 夏染染纤影怔了瞬，旋即气急败坏的看着他，“顾夜琛，你不要太过分了！”\n 幽冷的目光，居高临下的看着女孩不断往车边缩，澄澈眸子里闪烁的光影，有几分楚楚动人的意味。\n 仅看了她一眼，顾夜琛发动引擎，开走了车。目光一直看着前方，脸上再没有任何表情，更没有要放走她的意思。\n 车子一路驶入顾夜琛的私人豪宅，那个叫滨江湾的地方，湖光山水，波光粼粼的临水别墅，这个地方的确很漂亮，可也意味着夏染染想逃跑无望了。\n 之前回夏家，她都是走了一个小时才看到车，还遇到个色痞，要不是慕修敛，不对，慕修敛怎么会出现在这里？\n 而且她没有记错的话，他的车当时就是朝这方开的，是因为遇到她才调转了车头。\n 这里只有这一条路，而且路的尽头就是这一片的别墅群，那么她没有猜错的话，慕修敛应该也住在附近。\n 顾夜琛打开车门就看到女孩一双眼睛滴溜溜的打着转，就知道她肯定没有打什么好主意，有些不屑的扫了她一眼，唇角轻勾，淡淡的道，“夏染染，不想惹怒我，就给我安分守己一点，你知道我脾气不怎么好。”\n 岂止是不怎么好，是非常的不好！\n 就连这么云淡风轻的说话，也流露着一股子威胁的味道，夏染染看着男人俊美得让人恶寒的脸，磨磨蹭蹭的去解安全带。\n “要我帮你？”\n 夏染染缩在座位里，捏着自己受伤的掌心，眼睛扑闪扑闪，有些倔强的道，“不用。”她怎么觉得每次在顾夜琛面前，她就像个废物一样，什么事情都做不好，她很讨厌这种感觉，这种什么都要依赖别人的感觉。\n 她现在已经不适合依赖别人了，也没有资格。\n 顾夜琛闻言脸色却是暗沉了几分，伸出手臂就把她抱下车，夏染染不爽极了，“顾夜琛你听不懂人话吗？我都说了不用。”\n “我真的对你太仁慈了！”要不是念着她身上有伤，就凭她好几次顶撞惹怒他，他都想好好教训她一番。\n 可是这个不知死活的女人，他越是迁就忍耐她，她就越肆无忌惮，口无遮拦，就别再怪他不懂怜香惜玉。\n 夏染染看着男人那张漆黑如墨的脸，和稳健快速的迈步，顿时脊背寒凉，上一次他就是在她发烧的情况下强行要了她，把她做到了医院。\n 男人都是下半身动物，一旦兽性大发，不保准，他就不会在她满身伤痕时要了她！'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(0, result)

    def test_no33(self):
        textmod={'p':'顾夜琛却难得的没有动怒，伟岸身躯虚倚着餐桌，菲薄的唇瓣轻轻抿着，眸光轻扫，若有似无的打量着她。\n 那双鹰眸里迸发出的光芒，毫不掩饰的倨傲狂妄，夏染染觉得自己就像只被野兽瞄准的猎物，偏偏她还没有丝毫能够反抗的机会。\n 她咬唇，“尽管顾少长得秀色可餐，但还是麻烦您站远一点，不然这饭没法好好吃了。”就算逃不过那一劫，也要让她先吃饱吧。\n “夏染染，你在勾引我。”还没等夏染染反应过来他的话，他就抬手扣住她精美的下巴，俯首深深的吻住了她粉薄的唇瓣。\n 似乎是留意着她身上有伤，他的力道并不大，也灵巧的扣住她，让她不能动弹，软软甜甜的唇，快要沦陷他的神经。\n 明明只是想惩罚一下她的伶牙俐齿，可一碰到她，他脑海里那根理智的弦轻易就崩断，让他只想索取得更多。\n 刚浮现出这样想着，他已经伸手把人从椅子上抱起来，往楼上走去。\n “顾混蛋，你放开唔……”夏染染整个人慌乱又羞愤，两只爪子拼命的捶打着他健硕的胸膛，强烈的表示不满。\n 他怎么不去死！\n 吐出的字眼，似乎都夹带着滚烫的温度，“不想你的伤口裂开，就别乱动，乖乖配合我，我一定不让你痛。”\n 说完卧室的门就被重重的关上了。\n 陈妈走出厨房就看到这一幕，赶忙瑟缩着往后退了一步，呃，她什么都没有看到。\n 等夏染染睡醒的时候，都是第二天清晨了，阳光透过窗撒进屋里，有着温馨唯美的画面，一如既往，男人早早就起床离开了。\n 身侧的床褥里还残留着独属他的幽冷气息，似是不喜欢这种味道，夏染染不悦的蹙起了秀眉，立刻就要起床。\n 身子才动一下，骨头似乎被拆了的无力感瞬间袭来，她又重新倒回被窝里，由于动作幅度过大，牵扯着她浑身的伤口刺痛，她不由倒抽一口凉气。\n 顾夜琛那混蛋一定是野兽本性，凡是他想要的，不管别人是不是能承受，他都会肆无忌惮的夺取。\n 根本不管别人的死活！\n “人渣，王八蛋，我咒你一辈子不举！”眨巴掉眼睛里的雾水，夏染染吸了吸鼻子，缓缓下了床走到洗漱间。\n 于绯说她的伤口不能见水，可她不喜欢身上残留着的男人味道，她还是小心翼翼的脱掉衣服，用湿毛巾擦拭身体。\n 镜中的自己，雪白肌肤上除了鞭痕，再没有其它东西，她是不是应该庆幸，就算顾夜琛昨天兽性大发，他都没有对她动粗。\n 略带冰凉的小手缓缓覆盖住她脸颊的伤口，夏染染看着镜中那张惨白的小脸，指尖滑过已经结疤的红痕，有些膈手。\n 她眯了眯眼睛，声音凉凉的道，“夏延封，总有一天，你对我做的这些，我统统都会还到你女儿那张引以为傲的脸上。'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(0, result)

    def test_yes34(self):
        textmod={'p':'　　伴随着持续的捅插耸刺，朱隶逐渐发动魔功，胯下承受着强力淫奸的女人渐渐的哭尽甘来，当处女的血混合了溢出的淫水，她也开始体会到了性爱的舒爽'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes35(self):
        textmod={'p':'意飞神驰，朱隶一时陶醉在美女的羞怯和生疏中，那壮硕的阳具却不甘寂寞地暴涨起来，大有劈波斩浪之势，可惜浸在水里的下半身被浴桶形成的阴影遮住，只有那巨大的龙头在水波荡漾中忽隐忽现'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_no36(self):
        textmod={'p':'　　已是子夜时分，王府书房中却坐着七八人，气氛凝重'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(0, result)

    def test_yes37(self):
        textmod={'p':'　　云裳要害受袭，加上耳垂处传来的要命的热痒，芳心被激起了滔天的涟漪，美目紧闭，俏脸上潮红斑斑，樱唇中动人的呻吟声声不绝'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes38(self):
        textmod={'p':'　　她闭上双目等待着朱隶的暴风烈雨，从乳房上传来的刺痛使她秀眉微微的邹起，但是她没有反抗，因为现在在她上面的男人是她一生的等候'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes39(self):
        textmod={'p':'随着朱隶猛烈的冲击挺动，云裳觉得自己就像是要飞上天的那样欢快，她花蕊处的嫩肉伴随着朱隶的挺动而不断的翻转，带着纯阴气息幽香淡淡的蜜汁如珍珠一般缀满了芳草就好像是清晨的露珠点缀在草丛之间处处写满了生命的气息'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes40(self):
        textmod={'p':'的一声娇呼，云裳只觉得自己的灵魂完全不受肉体的束缚正无拘无束的漂游在天空，一阵极其猛烈的奇妙感觉从花蕊处升起，终于忍耐不得极度的享受，一股蕴涵着无数生命因子的赤纯元阴从花蕊深处迸射而出'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes41(self):
        textmod={'p':'半年来，虽时常将此女招入房中大逞手足之欲，但却一直没有进一步的行动'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes42(self):
        textmod={'p':'尽情的享用一番玉娟的处子之体'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes43(self):
        textmod={'p':'朱隶目不转睛的看着这具让人血脉贲张的胴体，心跳不由加速'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes44(self):
        textmod={'p':' 抱住玉娟微微颤抖的身体，朱隶直接一挺分身，'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes45(self):
        textmod={'p':'爽，实在是太爽了，仿佛能够感觉到玉娟那娇美的肉'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes46(self):
        textmod={'p':'　　平静下来后，恢复清醒的玉娟仍然紧紧将朱隶抱住不愿松开，柔软挺拔的双峰也紧紧贴住朱隶的胸膛，朱隶轻轻挣脱了她的怀抱抬起身来，床单上的一片落红映入眼帘，朱隶替一丝不挂的玉娟盖上被子，然后轻轻将其拥入怀中，道：'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes47(self):
        textmod={'p':'接着熟练的褪去了沈丽蓉的衣裙，但见这具身体虽然还略显青涩，但是匀称的身体，傲然挺拔的双峰，修长的双腿紧紧闭合，细腻的肌肤，雪白的颜色能让人看见细细的血管，微微隆起的小山丘上稀稀落落的点缀着几棵小草，粉红的禁地，只能让人略窥一点美景，更是引人瑕思'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes48(self):
        textmod={'p':'双手轻握小蛮腰，美丽的躯体骄傲的展现在面前，挺立的玉峰上一点诱人的嫣红，在嘴中娇艳欲滴，滚圆的臀部在手中不断的扭动'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes49(self):
        textmod={'p':'女孩子头一遭侍侯男人行房，破身子的时候，都会很疼的，忍忍吧'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes50(self):
        textmod={'p':'腰一挺，粗长硬挺的龙具全数没入了这个包夹紧绕着分身的温暖的甬道之中，但是娇嫩小巧的通道却容不下如此雄伟的凶器，不得已的开放了未知的禁地，朱隶感觉龙具坚硬火烫的头部突破了一个肉紧的关口，进入了愈加紧缠的狭小空间，嫩肉仿佛在那一瞬间都全部痉挛了起来，气势汹汹的挤压着可怜的分身，长吐了一口气，顶着尽头的肉壁划起了圆圈，'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes51(self):
        textmod={'p':'　　苏语蝶穿了一件水绿色的纱裙，薄如蝉翼，里面竟未穿亵衣，丰腴的胴体隐约可见'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes52(self):
        textmod={'p':'尤其是领口处开得甚低，直到腰际，弯腰撩水之时，前胸的纱衣垂了下来，正好让偷窥的朱隶清晰的看见两只圆润白皙的椒乳，随着主人的动作晃得人耳红心跳'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes53(self):
        textmod={'p':'朱隶轻笑着假装跌翻，倒在了还喘息不止的丽蓉、语蝶柔软的酥胸上，两女花容失色地合力将朱隶推了开去，齐齐娇嗔道：'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes54(self):
        textmod={'p':'　　暗淡的月光下，映照出地上一具雪白动人的美妙胴体，在光影中摇曳，清纯典雅的脸上带着融合了羞耻与悲愤的表情，全身上下仅余一条紫色的亵衣，娇美的躯体轻轻颤抖，在艳丽中又带着惹人怜惜的风情'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_no55(self):
        textmod={'p':'　　女人在火热的视线侵犯下，白晰的俏脸上布满了红晕，修长的眼睫毛上下轻颤，显示出其内心的激动'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(0, result)

    def test_no56(self):
        textmod={'p':'没想到我叶继欢梦寐以求的美人，竟然真在今天给我得尝宿愿，上天也算待我不薄了'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(0, result)

    def test_no57(self):
        textmod={'p':'咱们还是趁着现在找个地方先在两个丫头的身上好好快活快活，等回到咱百刀门，只怕用不了三天这两个丫头就给那帮爷们玩残了'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(0, result)

    def test_no58(self):
        textmod={'p':'我说这些，是要你们明白，加入了我朱隶争雄天下的计划，你们就不再是一个武林人了，必须放下门派之见，学会从天下大局的角度去看待这世间的一切'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(0, result)

    def test_no59(self):
        textmod={'p':'强者有善恶之分，弱者亦有善恶之别，而最高统治者的责任就是让强者缔造的秩序符合仁善的准则，让弱者的行为遵守仁善的准则，如此，天下就会太平兴盛'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(0, result)

    def test_yes60(self):
        textmod = {
            'p': '两个人说着，关了门，脱了裤子，靠在门后边干了起来。弟媳妇的大屁股墩在门上，杨主任一手搂着她的头，一手掂着鸡巴往她的水门里边送，刚刚深入进去，还没有活动，就听到院子里有人叫杨小三的名字。弟媳妇说：“是我娘来了，你赶紧拔出来算了，叫他看见不好。”'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes61(self):
        textmod = {
            'p': ' 男人和女人之间的激情碰撞，有时候就是这样自然而然，我们在床上激情地拥吻，她的手勾着我的脖子，主动伸出舌头与我纠缠，两年没碰女人的我很快就感到身体紧绷，我飞快地脱掉她的衣服，看着她粉色的胸罩底裤，只觉口干舌燥。但我没有猴急，慢慢引导着她。一室旖旎，可正当我做足前戏，准备进入的时候，门突然响了起来，我们都被吓停了动作，才发现是我儿子醒了，在敲门想要进来，她当时吓得眼睛大大的，看起来有些好笑跟可爱，可是已经到那地步了，临阵退缩我没办法，所以我决定继续，腰身一挺就占有了她，她忍不住叫了一声，可很快又害怕被听到动静似地用手捂住了嘴，她的身体太暖太软，已被情欲控制的我很快就无意识地开始折磨她'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes62(self):
        textmod = {
            'p': ' 进了房间，我被他抱在床上，我不敢去想我们明天会是怎样，只想放松自己，去感受他的爱，因为是第一次跟老公以外的男人，我有点紧张，他温柔地解我上衣的扣子，叫我别害怕，后来我们光着身子躺在床上，他亲吻我，抚摸我，每一个动作都让我融化在他的引导下，我的欲望一点一点地膨胀开来，开始迎合他，直到迎来了高潮'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes63(self):
        textmod = {
            'p': ' 一进门，我和她就紧紧地拥抱在一起，脱了她的衣服，触摸她的肌肤，很光滑，身上凉凉的，当我皮肤贴着她的皮肤，坦诚相待的那一刻，体验到一种从未有过的新鲜和刺激，那种肌肤之亲所产生的凉凉的性感，到现在都铭刻在我的记忆深处。我们不停地接吻，我的双手不停地游走，脸颊、耳根，脖子、锁骨、乳房、腹部、大腿内侧，一系列前戏下，她的下面早已水汪汪的一片，双眼微闭，愉悦地呻吟。'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes64(self):
        textmod = {
            'p': '一次次撞击，让卢慧慧欲罢不能，差点叫出声来。幸好张良发现得快，直接将衣服塞在了她的嘴中。'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes65(self):
        textmod = {
            'p': '见到时机已到，张良也不再做前戏，腰带一拉，褪下裤子。张良被王静紧紧的包裹住，开始缓缓的活动。顺手把王静的短裙直接往上一翻，身体一挺，直接冲了进去。'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes66(self):
        textmod = {
            'p': '张良知道这是到了最后的冲锋时期了，速度开始不断的加快，频率也不断的升高。-啊...啊…-王静被这突如其来的频率又送上了另一个高潮。'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes67(self):
        textmod = {
            'p': '忽然，感觉下面被两瓣樱唇裹住，里面还有一条舌头来回游动，不断的刺激着张良敏感的神经。樱唇慢慢将整个棍状物包了进去，也是陈惠的喉咙深，竟然全部裹了进去，不过最后还是顶在了喉咙深处。一股异样的感觉传遍了全身，陈惠的头一高一低，吸允着。速度也越来越快，在陈惠多方面的夹击之下，张良终于有了快要缴械的趋势。陈惠也似乎感觉到了，吸允的力气越来越大，速度也越来越快。'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes68(self):
        textmod = {
            'p': '已经干渴了好几天的张良已经有些迫不及待了，直接将陈娴的腰带打开，将她托在了病床上，迅速把裤子褪去。陈娴穿的粉色的内内，蕾丝的，隐隐看见里面黑色的绒毛。张良此时的下体早已坚硬似铁，大脑袋泛着紫色的光晕，一跳一跳的似乎在寻找着目标。'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes69(self):
        textmod = {
            'p': '好紧！这是张良的第一感觉，心想，这小妮子不会还是处女吧，但是这个想法还没有被否认的时候，就感觉到‘脑袋’前方碰到了一道屏障。咯噔！张良心里一跳，真是处！？再看陈娴，双眼紧闭，紧张的等待着，俨然一副要上刑场的样子，看在张良眼里别有一番风味，不过他眼神里露出了一丝温柔。俯下身去，吻在了陈娴紧张的脸上，下身也慢慢继续前挺。'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes70(self):
        textmod = {
            'p': '终于，两人在不断的抖动中结束了这次紧张而又刺激的活塞运动，一股滚烫的生命精华射在了王静身体深处，滋润着她的身心。'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes71(self):
        textmod = {
            'p': '张良由不得陈娴准备，扬起坚硬的长枪，瞄准洞口就刺了进去。还在睡梦中的陈娴，被突如其来的攻击吓了一跳，不过很快就在连续蠕动的快感中迷失了，女人，是永远都不会满足的动物。'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes72(self):
        textmod = {
            'p': '当一根根又粗又硬的黑棍子捅入他们的后庭花的时候，他们的世界仿佛一瞬间都崩塌了。吴小江终于知道什么叫做后悔了，他现在就极度的后悔，可是后悔又有什么用？现在菊花已经不抱，想到刚才这三个又黑又丑的大汉在自己身后蠕动，内心就闪过一丝恐惧。'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes73(self):
        textmod = {
            'p': '张良见陈娴不说话，直接将她身体压低，不顾现在的场合，直接拉开裤链，将自己的命根子释放出来，此时他已经是雄赳赳气昂昂，整装待发，还一抖一抖的，仿佛已经忍不住要进攻了似的。陈娴见到后，忍不住的喜欢，就是这个东西不止一次的让自己尖叫，又让自己爱不释手，她时常在想，为什么男人和女人的差距怎么就这么大，只是因为一个带吧，一个不带把，一个多两坨肉，一个少两坨肉吗？'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes73(self):
        textmod = {
            'p': '张良探索她身体的时候，陈娴的手已经抓在了那个火烫的肉柱上，上下来回的套弄着。张良已经忍不住了，再加上环境特殊，张良直接一把抱起陈娴，有些粗暴的将她的腰带解下，将她的裤子稍微褪下去一点，但是不太明显。张良直接扒下陈娴的内裤，将坚挺依旧的东西一下插进了空虚寂寞的洞口。没有起初的疼痛感，有的只是不停来回的爽快和前所未有的刺激。'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes74(self):
        textmod = {
            'p': '张良和陈娴都有些呆滞的看着突然出现的李军，顿时有些尴尬，此时张良的下身还留在陈娴体内，抵达花蕊的最深处，但是突然出现的情况让他们不敢轻举妄动。'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes75(self):
        textmod = {
            'p': '经过李军这么一惊吓，张良和陈娴草草完事，张良帮陈娴擦拭下身，又是一顿捅，惹得陈娴就那么站着，还要表现出很正经的样子，因为她站起来的时候，其他人是能看见她的。但是这么正经的表情下面，却有一双邪恶的大手在无情的侵袭着。到陈娴第二次高潮的时候，陈娴实在忍不住下身传来的刺激，趴在桌子上，捂着嘴，努力不让自己发出羞涩的声音。'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes76(self):
        textmod = {
            'p': '她的头向下移去，目标已经非常明显。舌头已经向龙出洞穴一般缠向了枪头上，不断的吸允着，似乎能从那里索取来玉液琼浆一般。张良这是第二次享受这种待遇，上一次是在医院，对方是陈惠。不同于陈惠的迅疾，妖龙慢慢的用手把住枪杆，细细品尝，带给张良的是一股异样的刺激。'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes77(self):
        textmod = {
            'p': ' 胡媚子说：“你难道不想吗，自己看看吧，鸡巴上都湿哒哒的啦。”'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes78(self):
        textmod = {
            'p': '张良手把陈娴外衣的扣子解开，露出粉色的毛衫，毛衫的质地很软，张良从毛衫的下摆伸了进去，这次他省去了所有的前戏，直接向那最高峰进攻过去，张良凭借着自己娴熟的经验，毫不费力的就把陈娴那藏有数层机关的胸罩解开了。终于把陈娴释放出来了，陈娴的两颗粉馒头在失去胸罩的保护下，直接和毛衫亲密接触在一起，有种痒痒的感觉，而且那只大手还在不停的揉捏着，给自己的身体竟然带来异样的快感，这是以前从没有过的。'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes79(self):
        textmod = {
            'p': '而且峰顶的紫葡萄在毛衫来回摩擦的时候产生的那种揪心的感觉，更让她呼吸急促，不能自已。张良手下不断的侵袭着陈娴的神经，不停的将两个大蜜桃摆弄出不同的形状，每次这种时候，陈娴都会发出诱人的细叫，惹的张良加大手上的力度。'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes80(self):
        textmod = {
            'p': '看着两颗大白兔在眼前一跳一跳的，张良直接像饿虎扑食一般用嘴疯狂的吸允了上去，不断地咬着，含着。张良忘我的吸允着。嘴上含着陈娴的小紫葡萄，舌头在上面来回的弄着，惹得陈娴心里难耐不已，手已经向陈娴腰带解去，陈娴下面仅仅穿了一条深蓝色的警裤，解开腰带后，张良轻轻把陈娴托起来就把裤子拨了下去。今天陈娴穿的是粉色的小裤裤，还是半透明的，从张良的角度看，能看到里面半遮半露的风景。张良在将陈娴挑拨上一个高峰的时候，果断的放弃了双峰，向下吻去，不一会就游走到了陈娴肚脐处陈娴突然捂住自己的身体，媚眼如丝的看着张良，摇了摇头，用细若游丝的声音说道：不要，脏！张良温柔的笑笑，轻轻的拿开陈娴的手，接着吻了下去。这里的每一寸肌肤细腻温润，张良当然不会放过。而陈娴此时完全是被张良弄的浑身乏力，只用两条雪白的胳膊向后撑着办公桌，嗓子里不停的发出诱人的轻声。而且声音还在逐渐的扩大，不是她有意为之，完全是因为她控制不住自己的叫声。想到马上就要享受到无尽的快感的时候，陈娴不禁迷醉在了下沉传来的快感之中'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes81(self):
        textmod = {
            'p': '终于姬雪的手向张良的下面摸去，那里的东西是她喜爱的东西，也是和她唯一不一样的地方。轻轻的解开了张良的系着的绳子，将张良的裤衩向下褪了下去，手就摸到了张良那上面。'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes82(self):
        textmod = {
            'p': '张良的手自然也不会驻足于姬雪的后背，一手搓揉着她的臀，另一只手已经来到前方，是该释放一下挤压在胸膛上的双峰了。当张良攀上那双峰时，单手用力，将那对巨大的大白兔揉成各种形状，嘴还是和姬雪激情的吻着。另一种抚着姬雪翘臀的手慢慢绕过侧面，来到前方，这里才是张良最后要攻城略地的地方。'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes83(self):
        textmod = {
            'p': '张良手支撑在浴缸的两边，压在姬雪的身上，姬雪仰躺在浴缸里，媚眼如丝的看着张良的下面，向往的神色一览无余。既然姬雪索取了，张良是不会让他失望的好紧！这是张良的感觉，这几个月，张良一直都是和王静和陈娴，虽然两人都不可多得的美女，但是早已被张良开垦肥沃了，已经不用施肥就能长出好庄稼了。'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes84(self):
        textmod = {
            'p': '王静感受到下面那滚烫的手，和张良前面已经微微隆起的小帐篷，嘴角露出一个妩媚的笑，直接贴了上去，胸前的两对胸器静静的贴在张良的胸膛上。'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes85(self):
        textmod = {
            'p': '当张良不满足于这一层淡淡的纱衣，将它轻轻的脱离了王静的身体，当王静的所有展现在张良面前时，张良不由的摇头感叹，那胸前的凶器能秒杀掉多少英雄好汉啊，我张良何德何能能占据这样两座雄伟的山峰。那双峰间是一条深不见底的深沟，张良二话不说，直接一头扎进那道深深的沟壑里，两边的峰壁不断地磨蹭着张良的脸侧，一种说不出的感觉油然而生，张良终于将自己埋藏在了谷底，触碰到了谷底那狭窄的通路。还不时的吻向旁边的峰壁，有时候这种被打入深渊的感觉要比攀上高峰更有成就感。当张良再一次从谷底攀上峰顶时，大起大落的心情让张良大呼过瘾。'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes86(self):
        textmod = {
            'p': '陈娴闭着眼睛享受着身体上带来的快感，她的脸贴在桌子上，身体摇晃着。但是她的心里充满了甜蜜。张良每一次的运动，她的身体就传来一种达至巅峰的舒服，就像得到了时间最大的满足似的，没有什么能够阻挡她要寻求快乐的心。'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes87(self):
        textmod = {
            'p': '张良的速度与频率越来越快，也在慢慢的带动陈娴走向高峰，每一次来回都是两人最深处的交流，陈娴把自己的一切都交给了张良，张良也将自己的身心交给了陈娴。'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes88(self):
        textmod = {
            'p': '强势而暴力的掰开陈娴的膝盖，也轻轻的将她的粉色小蕾丝褪到了膝盖处，而张良直接摇晃着舌头，冲进了那深处。'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes89(self):
        textmod = {
            'p': '陈娴的身体轻微的颤抖着享受着。陈娴弯曲这双膝，两只脚丫子踩在桌沿上，她仰躺在桌子上，两个坚挺而高大的山峰全部堆在了她的身体上，依旧是那么坚挺，没有一些女人一样堆在那里想一堆肥肉似的瘫在那里。陈娴的身体忍不住的晃动，办公室的空气中弥漫着欢爱的气息。'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes90(self):
        textmod = {
            'p': '张良知道是时候发起进攻了，陈娴在下面舔舐的他已经有了要缴械的冲动了，不得不说嘴和身体的其他部分就是不一样。'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes91(self):
        textmod = {
            'p': '王静抓过张良的下面，直接含进了嘴里，她那吊着的木瓜形状的胸碰到张良的大腿上，更是给他一种异样的感受。'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes92(self):
        textmod = {
            'p': '随着频率的加快，张良没入的深度也在逐渐的加深，一直深到柯思雨可以承受的极限！张良就是要保持这种逐渐被填满的满足感，让柯思雨永远不知道张良的货真价实。'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes93(self):
        textmod = {
            'p': '张良翻过柯思雨的身体，将她的身体直直的贴在她身后的落地玻璃上，屁股都被挤的变了形，张良手摸向柯思雨雪白的臀部，并且嘴上已经攀上了柯思雨的略带秀气的胸脯，虽然她的胸没有陈娴的硕大，也没有王静的坚挺，但是在张良看来确实刚刚好，一只手刚好能抓住大半个，让张良有种一切竟在手中的掌控感。'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes94(self):
        textmod = {
            'p': '张良吻得喜不自胜，手上已经隔着里面穿的线衣在姬雪的高傲的胸上来回的搓揉，不得不说，姬雪的胸属于超柔软的，张良无论揉捏成任何形状，最后都能变回原来的样子。'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes95(self):
        textmod = {
            'p': '张良很坐在姬雪的身体上，姬雪的面前就是张良的内裤，她看的喜爱，直接张开嘴没想到姬雪会为自己带来这样的待遇，自然是心下欢喜。'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes96(self):
        textmod = {
            'p': '此时，张良已经在姬雪的大腿附近游弋了一圈，然后从姬雪的内裤上面伸了进去姬雪轻轻的发出了舒服的叫声。细微的叫唤不禁让张良血脉喷张，张良的手指不禁再一次的加大了力度和加快了速度。啊！'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes97(self):
        textmod = {
            'p': '随后，张良才将姬雪的内裤褪了下去，露出了里面的庐山真面目。张良伸出舌头，直接舔舐了过去。'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes98(self):
        textmod = {
            'p': '张良看准时机，直接一个金枪突刺。本就想给姬雪刺激的张良，没有选择逐步加深，而是猛力进攻。姬雪还根本没有适应张良的身体，在张良忽然猛力进攻的时候，除了浑身撕裂一般的疼痛之外，还有一种直抵深处的说不出的畅快。'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes99(self):
        textmod = {
            'p': '王静在看到张良的金枪的时候，眼睛不禁又是一亮，不知道是不是错觉，我感觉张良的下面似乎又粗大了几分，但是如果细看的死话似乎又和从前一样，当然眼睛看到的都是虚的，只有当进入自己身体检验的时候才能真正的检验出来是否是错觉。王静看到让他爱不释手的金枪，在手上玩着，争取让张良感受到最舒适的体验。'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes100(self):
        textmod = {
            'p': '陈娴忍不住轻叫了一声，看来也是很久没有受到张良的关怀了，这才忍不住叫了出来。王静在下面也没有闲着，纤细的手指摆弄着张良的金枪，不得不说，现在王静的手艺越来越好了，能将张良的金枪肆意的摆弄，就像世界上最好玩的玩具似的。'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes101(self):
        textmod = {
            'p': '玩弄了一会，王静也觉得不够满足，她的舌头先伸了出来，上面那股男人特有的气味，总是能让她们这些女人们发狂。张良忽然感觉浑身的细胞都被激活，而且每一寸肌肤都在欢愉，汗毛都根根竖立。'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes102(self):
        textmod = {
            'p': '张良也知道是时机发起总攻了，张良站起身来，把已经脱的精光的王静反放在沙发上，让她趴在那里。嗤！张良的金枪再一次故地重游，但是很显然每一次他都能有不一样的体验。'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes103(self):
        textmod = {
            'p': '这时张良的手指渐渐地到了柯思雨的身体里，那是一种淡淡的感觉，但是却带动的是全身血液的流通，柯思雨感觉自己已经舒服的不能自已了，尤其是在张良的身体上坐着，手里还握着张良的那里。张良的手速度加快，而柯思雨的身体就随着这样的旋律，而且身体里传来的快感让她觉的幸福极了。很快，张良感觉到紧紧的，知道柯思雨的第一次就要来了，手上的动作不禁更加快了。'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes104(self):
        textmod = {
            'p': '张良亲吻着蓝月的身体，熟练的将她的文胸接下，两只大白兔跃然出现在眼前，在解开的瞬间还颤了两下，似乎在向张良打招呼。张良大嘴一张，含住蓝月的一支巨峰，舌头还在不断的舔舐，让蓝月的身体都陷入了一种奇妙的状态。'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes105(self):
        textmod = {
            'p': '在胡思乱想的时候，张良终于触摸到了蓝月的大腿内侧。手上传来温润的触感，看来蓝月已经有些难耐了。张良轻轻的触了下门户，虽然没有进去，但却也感觉到了紧蹙，这是已经很久没有遇到过的了。'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes106(self):
        textmod = {
            'p': '张良的魔手渐渐褪去她的内裤，终于清楚看清了里面的场景。那处子幽香，不禁让张良有些神魂颠倒。'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes107(self):
        textmod = {
            'p': '他忽然改变了主意，张良直接跨坐在蓝月的身上，立起身子，快速的脱掉自己的衣服，裤子，裤衩已然遮不住金枪的规模。好大啊！'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)

    def test_yes108(self):
        textmod = {
            'p': '终于，张良在紧张的氛围之下就要释放了自己。在一阵疯狂的加速之后，张良和蓝月双双缴械，两个人紧紧的抱在一起，蓝月长长的指甲掐着张良的后背，或许是疼的缘故，他们都没有说话。'}
        result = int(self.sendRequest(textmod))
        self.assertEqual(1, result)
if __name__ == '__main__':
    unittest.main()
